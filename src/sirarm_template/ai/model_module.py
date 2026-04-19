import datetime
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from sirarm_template.utils.ops import parse_version
from sirarm_template.utils.torch import get_grad_scaler, load_checkpoint_support_submodule, is_parallel_model
from sirarm_utils import increment_path
from sirarm_utils.logger import setup_logger


class PARALLEL_MODE(Enum):
    CPU = "CPU"
    SS_GPU = "SS_GPU"
    SM_GPU_DP = "SM_GPU_DP"
    SM_GPU_DDP = "SM_GPU_DDP"
    MM_GPU = "MM_GPU"


class DATASET_MODE(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


MODE_TIP_MAP = {
    "en": {
        "CPU": "CPU",
        "SS_GPU": "Single-Machine-Single-GPU",
        "SM_GPU_DP": "Single-Machine-Multi-GPU using DP(torch.nn.DataParallel)",
        "SM_GPU_DDP": "Single-Machine-Multi-GPU using DDP(torch.nn.parallel.DistributedDataParallel)",
        "MM_GPU": "Multi-Machine-Multi-GPU",
    },
    "zh": {
        "CPU": "CPU模式",
        "SS_GPU": "单机单GPU模式",
        "SM_GPU_DP": "使用 DP（torch.nn.DataParallel）的单机多GPU模式",
        "SM_GPU_DDP": "使用 DDP（torch.nn.parallel.DistributedDataParallel）的单机多GPU模式",
        "MM_GPU": "多机多GPU模式",
    }
}


class ModelModule(ABC):
    def __init__(
            self,
            config: dict,
            logger_name_prefix="ModelModule",
            running_path="./runs/exp",
            running_path_increment=True,
            use_amp=False,
            use_dp=False,
            use_parallel=True,
            parallel_backend="gloo",
            show_running_info=True,
            **kwargs,
    ):
        self.init_kwargs = kwargs
        # init status param | 初始化状态参数
        self.cuda_available = None
        self.world_size = None
        self.rank = None
        self.local_rank = None
        self.device_count = None
        self.is_parallel = None
        self.is_master = None

        self.device = torch.device("cpu")
        self.early_stop_counter = 0
        self.early_stop_patience = 10
        self.best_metrics = None
        self.epochs = 100
        self.current_epoch = 0
        self.save_freq = 10
        self.is_training = True

        self.config = config
        self.use_amp = use_amp

        # 并行参数 | parallel param
        self.use_dp = use_dp
        self.use_parallel = use_parallel
        self.parallel_backend = parallel_backend
        self.parallel_mode = PARALLEL_MODE.CPU
        self._auto_init_parallel()
        # 运行路径 | running path
        self.running_path = self._init_running_path(running_path, running_path_increment)
        # 日志初始化 | logger initialization
        self.logger_name_prefix = logger_name_prefix
        # 控制台日志
        self.logger = (
            setup_logger(name=self.logger_name_prefix)
            if not self.is_parallel
            else setup_logger(f"{self.logger_name_prefix} rank_{self.rank}")
        )
        # tensorboard 日志初始化 | tensorboard logger initialization
        self.tb_logger = SummaryWriter(log_dir=str(Path(self.running_path) / "tb_log"))

        self.dataloader_config = {
            "train": {},
            "val": {},
            "test": {},
        }
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = get_grad_scaler() if self.cuda_available else None

        # 运行信息输出 | print running info
        if show_running_info:
            info = self.running_info()
            self.logger.info(info["info_log"])

    def running_info(self):
        return {
            "parallel_mode": self.parallel_mode.value,
            "device": str(self.device),
            "running_path": str(self.running_path),
            "config": self.config,
            "world_rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "info_log": "\n当前运行模式：{}\nCurrent running mode: {}\n当前运行设备(Current running device)：{}\n当前运行路径(Current running path)：{}\n当前运行配置为(Current running config)：{}\n".format(MODE_TIP_MAP["zh"][self.parallel_mode.value], MODE_TIP_MAP["en"][self.parallel_mode.value], self.device, self.running_path, self.config, )
        }

    def _broadcast_safe_obj(self, obj, use_cuda=False):
        if isinstance(obj, torch.Tensor):
            if use_cuda:
                obj.to(self.device)
                return obj
            else:
                return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: self._broadcast_safe_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._broadcast_safe_obj(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._broadcast_safe_obj(v) for v in obj)
        return obj

    def sent_broadcast(self, content, src=0):
        payload = self._broadcast_safe_obj(content)
        if self.is_parallel and dist.is_available() and dist.is_initialized():
            object_list = [payload]
            dist.broadcast_object_list(object_list, src=src)
            return object_list[0]
        return payload

    def receive_broadcast(self, src=0):
        if self.is_parallel and dist.is_available() and dist.is_initialized():
            object_list = [None]
            dist.broadcast_object_list(object_list, src=src)
            return self._broadcast_safe_obj(object_list[0])
        return None

    def _init_distributed(self):
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    backend=self.parallel_backend,
                    timeout=datetime.timedelta(minutes=30),
                )
            except Exception as e:
                raise RuntimeError(
                    f"初始化分布式进程组失败 | Failed to initialize distributed process group \n {e}"
                )

    def _auto_init_parallel(self):
        # GPU可用 | GPU available
        self.cuda_available = torch.cuda.is_available()
        # 全局设备数量 | global device number
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        # 全局设备编号 | global device id
        self.rank = int(os.environ.get("RANK", -1))
        # 本地设备编号 | local device id
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # 本地设备数量 | local device number
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        # 是否并行 | is parallel
        self.is_parallel = self.world_size > 1
        # 是否为主进程 | is master process
        self.is_master = (self.rank == -1 or self.rank == 0)

        if (not self.use_parallel) or (not self.cuda_available):  # CPU 模式 | CPU mode
            self.parallel_mode = PARALLEL_MODE.CPU
            self.is_parallel = False
            self.rank = -1
            self.local_rank = 0
            self.device = torch.device("cpu")
        else:
            if 1 < self.device_count == self.world_size:  # 单机多卡(SM-GPU) | single machine multi GPU (SM-GPU)
                self.is_parallel = True
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                if not self.use_dp:
                    self.parallel_mode = PARALLEL_MODE.SM_GPU_DDP
                    self._init_distributed()
                else:
                    self.parallel_mode = PARALLEL_MODE.SM_GPU_DP

            elif self.world_size > self.device_count:  # 多机多卡(MM-GPU) | multi machine multi GPU (MM-GPU)
                self.parallel_mode = PARALLEL_MODE.MM_GPU
                self.is_parallel = True
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)
                self._init_distributed()

            else:  # 单机单卡(SS-GPU) | single machine single GPU (SS-GPU)
                self.parallel_mode = PARALLEL_MODE.SS_GPU
                self.is_parallel = False
                torch.cuda.set_device(self.local_rank)
                self.device = torch.device("cuda", self.local_rank)

    def _init_running_path(self, running_path, increment):
        if self.is_parallel:
            if self.is_master:
                base_running_path = increment_path(running_path, sep="_", mkdir=False, increment=increment)
                self.sent_broadcast(base_running_path)
            else:
                base_running_path = self.receive_broadcast()
        else:
            base_running_path = increment_path(running_path, sep="_", mkdir=False, increment=increment)

        if self.parallel_mode in [PARALLEL_MODE.SM_GPU_DDP, PARALLEL_MODE.MM_GPU]:
            return increment_path(
                Path(str(base_running_path)) / f"rank_{self.rank}",
                sep="_",
                mkdir=True,
                increment=False,
            )
        else:
            return increment_path(
                base_running_path,
                sep="_",
                mkdir=True,
                increment=False
            )

    def _validate(self, model, parallel=True, **kwargs):
        val_loader = self._load_dataloader(DATASET_MODE.VAL, parallel=parallel, **kwargs)
        model.eval()
        with torch.no_grad():
            metrics = self.evaluate_train(dataloader=val_loader, model=model, **kwargs)
        return metrics

    def _load_parallel_sampler(self, dataset, shuffle):
        if self.parallel_mode in [PARALLEL_MODE.SM_GPU_DDP, PARALLEL_MODE.MM_GPU]:
            return DistributedSampler(dataset, shuffle=shuffle, rank=self.rank, num_replicas=self.world_size)
        else:
            return None

    def _load_dataloader(self, mode: DATASET_MODE, parallel=False, **kwargs):
        if mode not in [DATASET_MODE.TRAIN, DATASET_MODE.VAL, DATASET_MODE.TEST]:
            raise ValueError(f"{mode}是不可识别的模式 | {mode} is not a valid mode")

        if mode == DATASET_MODE.TRAIN:
            dataset = self.load_dataset_train(**kwargs)
        elif mode == DATASET_MODE.VAL:
            dataset = self.load_dataset_val(**kwargs)
        else:
            dataset = self.load_dataset_test(**kwargs)
        dataloader_config = self.dataloader_config.get(str(mode.value), dict()).copy()
        shuffle = dataloader_config.get("shuffle", False)
        if parallel:
            sampler = self._load_parallel_sampler(dataset, shuffle)
        else:
            sampler = None
        if sampler is not None:
            dataloader_config["shuffle"] = False
            dataloader_config["sampler"] = sampler
            return DataLoader(dataset, **dataloader_config)
        else:
            return DataLoader(dataset, **dataloader_config)

    def model_parallelization(self, model, ddp_config=None):
        if ddp_config is None:
            ddp_config = {}
        if self.parallel_mode in [PARALLEL_MODE.CPU, PARALLEL_MODE.SS_GPU]:
            model = model.to(self.device)
        elif self.parallel_mode == PARALLEL_MODE.SM_GPU_DP:
            model = model.to(torch.device("cuda", 0))
            model = torch.nn.DataParallel(
                model,
                device_ids=list(range(self.device_count))
            )
        elif self.parallel_mode == PARALLEL_MODE.SM_GPU_DDP:
            model = model.to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                **ddp_config,
            )
        elif self.parallel_mode == PARALLEL_MODE.MM_GPU:
            model = model.to(self.device)
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                **ddp_config,
            )
        return model

    def set_dataloader_config(self, mode: DATASET_MODE = DATASET_MODE.TRAIN, **kwargs):
        if mode not in [DATASET_MODE.TRAIN, DATASET_MODE.VAL, DATASET_MODE.TEST]:
            raise ValueError(f"mode must be train/val/test, but got {mode}")
        config_dataloader = {}
        for k, v in kwargs.items():
            config_dataloader[k] = v
        self.dataloader_config[str(mode.value)] = config_dataloader

    def run_train(
            self,
            resume=None,
            weight=None,
            epochs: int = 100,
            early_stop_patience: int = 10,
            use_dp: bool = False,
            use_amp: bool = True,
            use_val: bool = True,
            use_best: bool = True,
            use_last: bool = True,
            use_freq: bool = False,
            save_freq: int = None,
            **kwargs
    ):
        self.epochs = epochs
        self.use_dp = use_dp
        self.use_amp = use_amp
        self.early_stop_patience = early_stop_patience
        self.save_freq = save_freq if save_freq is not None else self.save_freq

        self.is_training = True
        self.load_model(**kwargs)
        self.froze_model(**kwargs)
        self.model = self.model_parallelization(self.model)
        self.load_optimizer(**kwargs)
        self.load_scheduler(**kwargs)

        start_epoch = 0
        if weight:
            self.load_checkpoint(weight, is_load_optimizer=False, is_load_scheduler=False, **kwargs)
            self.current_epoch = 0
            if self.logger:
                self.logger.info(f"从 {weight} 加载权重开始训练 | Load {weight} to start training")

        if resume:
            self.load_checkpoint(resume, is_load_optimizer=True, is_load_scheduler=True, **kwargs)
            self.logger.info(
                f"Resuming training from `{resume}`: epoch {self.current_epoch + 1} from 1"
            )
            start_epoch = self.current_epoch + 1

        train_loader = self._load_dataloader(DATASET_MODE.TRAIN, parallel=True, **kwargs)
        early_stop_flag = False
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            self.cancel_froze_model(epoch, **kwargs)
            self.train_one_epoch(epoch, epochs, train_loader, self.model, **kwargs)

            metrics = self._validate(self.model, self.is_parallel, **kwargs) if use_val else {}
            if self.is_master:
                # 间隔保存模型 | Save model at intervals
                if use_freq:
                    if (epoch + 1) % self.save_freq == 0:
                        self.save_checkpoint(name=f"model_epoch_{epoch + 1}", **kwargs)
                # 通过验证判断保存模型 | Save model based on validation metrics
                if use_best and use_val:
                    if self.best_metrics is None:
                        self.best_metrics = metrics
                        self.early_stop_counter = 0
                    else:
                        if self.is_best_model(metrics, **kwargs):
                            self.best_metrics = metrics
                            self.early_stop_counter = 0
                            self.save_checkpoint(name="best", **kwargs)
                        else:
                            self.early_stop_counter += 1
                # 保存最新模型 | Save the last model
                if use_last:
                    self.save_checkpoint(name="last", **kwargs)
                self.custom_save(epoch, metrics, **kwargs)
                # 早停判断 | Early stopping check
                early_stop_flag = self.is_early_stopping(metrics=metrics, early_stop_count=self.early_stop_counter, **kwargs)
                if self.parallel_mode in [PARALLEL_MODE.SM_GPU_DDP, PARALLEL_MODE.MM_GPU]:
                    early_stop_tensor = torch.tensor(
                        [1 if early_stop_flag else 0], device=self.device
                    )
                    self.sent_broadcast(early_stop_tensor)
            else:
                if self.parallel_mode in [PARALLEL_MODE.SM_GPU_DDP, PARALLEL_MODE.MM_GPU]:
                    early_stop_tensor = self.receive_broadcast()
                    early_stop_flag = early_stop_tensor.item() == 1

            if early_stop_flag:
                break

    def run_val(self, **kwargs):
        assert self.model is not None, "未加载模型。请确保load_model() 方法对 self.model 赋值，并在已经调用 | The model is not loaded. Make sure that the load_model() method assigns a value to self.model and is already called"
        self.is_training = False
        model = self.model
        model.eval()

        with torch.no_grad():
            dataloader = self._load_dataloader(DATASET_MODE.VAL, parallel=False, **kwargs)
            return self.evaluate_val(dataloader, model, **kwargs)

    def run_infer(self, is_loader: bool = False, **kwargs):
        assert self.model is not None, "未加载模型。请确保load_model() 方法对 self.model 赋值，并在已经调用 | The model is not loaded. Make sure that the load_model() method assigns a value to self.model and is already called"
        self.is_training = False
        model = self.model
        model.eval()

        with torch.no_grad():
            if is_loader:
                dataloader = self._load_dataloader(DATASET_MODE.TEST, parallel=False, **kwargs)
                return self.interface_loader(dataloader, model, **kwargs)
            else:
                return self.interface_single(model, **kwargs)

    def froze_model(self, **kwargs):
        pass

    def cancel_froze_model(self, epoch, **kwargs):
        pass

    def load_optimizer(self, **kwargs):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def load_scheduler(self, **kwargs):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def update_scheduler(self, metrics, **kwargs):
        self.scheduler.step()

    def is_early_stopping(self, metrics, early_stop_count, **kwargs):
        return early_stop_count > self.early_stop_patience

    def is_best_model(self, metrics, **kwargs):
        if self.best_metrics["global"] > metrics["global"]:
            return True
        else:
            return False

    def load_checkpoint(self, path: str = None, is_load_optimizer: bool = True, is_load_scheduler: bool = True, strict: bool = True, model_param_name: str = "model", optimizer_param_name: str = "optimizer", scheduler_param_name: str = "scheduler", **kwargs):
        assert path is not None, "检查点/权重文件路径为空，无法加载 | The checkpoint / weight file path is empty and cannot be loaded"
        assert self.model is not None, "请先加载模型 | Please load the model first"
        if parse_version(torch.__version__) >= (2, 6, 0):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        else:
            checkpoint = torch.load(path, map_location=self.device)
        # load model state dict
        if is_parallel_model(self.model):
            self.model.module = load_checkpoint_support_submodule(
                model=self.model.module,
                checkpoint=checkpoint,
                param_name=model_param_name,
                strict=strict,
                sub_name=None,
                load_args={}
            )
        else:
            self.model = load_checkpoint_support_submodule(
                model=self.model,
                checkpoint=checkpoint,
                param_name=model_param_name,
                strict=strict,
                sub_name=None,
                load_args={}
            )
        # load optimizer state dict
        if is_load_optimizer and self.optimizer is not None and optimizer_param_name in checkpoint:
            self.optimizer.load_state_dict(checkpoint.get(optimizer_param_name))

        # load scheduler state dict
        if is_load_scheduler and self.scheduler is not None and scheduler_param_name in checkpoint:
            self.scheduler.load_state_dict(checkpoint.get(scheduler_param_name))

        self.current_epoch = checkpoint.get("epoch", 1) - 1
        if self.logger is not None:
            self.logger.info(
                f"Checkpoint loaded from {path};  save the weight time is {checkpoint.get('timestamp')};the epoch is {self.current_epoch + 1} from 1;"
            )

    def save_checkpoint(self, path=None, name="model", ext=".mmpt", model_param_name: str = "model", optimizer_param_name: str = "optimizer", scheduler_param_name: str = "scheduler", **kwargs):
        checkpoint = {
            "epoch": self.current_epoch + 1,
            model_param_name: (
                self.model.module.state_dict()
                if is_parallel_model(self.model)
                else self.model.state_dict()
            ),
            optimizer_param_name: self.optimizer.state_dict(),
            scheduler_param_name: (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {},
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
        if path is None:
            path = self.running_path
            Path(path).mkdir(parents=True, exist_ok=True)
            filepath = Path(path) / str(name + ext)
        else:
            filepath = Path(path) / str(name + ext)
        torch.save(checkpoint, filepath)
        if self.logger is not None:
            self.logger.info(f"Checkpoint saved to {filepath}")

    @abstractmethod
    def load_dataset_train(self, **kwargs):
        raise NotImplementedError(
            "load_dataset_train(self, **kwargs)方法必须实现 | load_dataset_train(self, **kwargs) method must be implemented"
        )

    @abstractmethod
    def load_model(self, **kwargs):
        raise NotImplementedError(
            "load_model(self, **kwargs)方法必须实现 | load_model(self, **kwargs) method must be implemented"
        )

    @abstractmethod
    def train_one_epoch(self, epoch, epoches, dataloader, model, **kwargs):
        raise NotImplementedError(
            "train_one_epoch(self, epoch, epoches, dataloader, model, **kwargs)方法必须实现 | train_one_epoch(self, epoch, epoches, dataloader, model, **kwargs) method must be implemented"
        )

    def evaluate(self, dataloader, model, **kwargs):
        raise NotImplementedError(
            "evaluate(self,dataloader, model, **kwargs)方法必须实现 | evaluate(self,dataloader, model, **kwargs) method must be implemented"
        )

    def evaluate_train(self, dataloader, model, **kwargs):
        self.logger.warning("目前已将训练时evaluate和验证时evaluate实现分离。建议实现`train_evaluate(self, dataloader, model, **kwargs)`方法来替代原evaluate。当前版本依旧兼容evaluate作为训练时evaluate，将在0.0.4版本移除支持。| Currently, the evaluation at training time and evaluate at validation time have been separated. It is recommended to implement the 'train_evaluate (self, dataloader, model, kwargs)' method to replace the original evaluate. The current version is still compatible with evaluate as a training evaluate, and will be removed in version `0.0.4`")
        return self.evaluate(dataloader, model, **kwargs)

    def evaluate_val(self, dataloader, model, **kwargs):
        raise NotImplementedError(
            "val_evaluate(self, dataloader, model, **kwargs)方法必须被实现 | val_evaluate(self, dataloader, model, **kwargs) method must be implemented"
        )

    def load_dataset_val(self, **kwargs):
        raise NotImplementedError(
            "load_dataset_val(self, **kwargs) 方法没有被实现 | load_dataset_val(self, **kwargs) method has not been implemented"
        )

    def load_dataset_test(self, **kwargs):
        raise NotImplementedError(
            "load_dataset_test(self, **kwargs) 方法没有被实现 | load_dataset_test(self, **kwargs) method has not been implemented"
        )

    def interface_loader(self, dataloader, model, **kwargs):
        raise NotImplementedError(
            "interface_loader(self, dataloader, model, **kwargs) 方法没有被实现 | interface_loader(self, dataloader, model, **kwargs) method has not been implemented"
        )

    def interface_single(self, model, **kwargs):
        raise NotImplementedError(
            "interface_single(self,model,**kwargs) 方法没有被实现 | interface_single(self,model,**kwargs) method has not been implemented"
        )

    def custom_save(self, epoch, metrics, **kwargs):
        pass
