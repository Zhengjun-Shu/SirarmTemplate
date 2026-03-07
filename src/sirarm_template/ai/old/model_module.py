import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sirarm_utils import increment_path
from sirarm_utils.logger import setup_logger


class ModelModule(ABC):
    def __init__(
        self,
        config: dict,
        early_stop_patience: int = 5,
        logger_name_prefix="ModelModule",
        running_path="./runs/exp",
        running_path_increment=True,
        use_dp=False,
        parallel_backend="gloo",
        **kwargs,
    ):
        self.logger_name_prefix = logger_name_prefix
        self.parallel_backend = parallel_backend
        # * parallel set
        # use ddp for parallel training
        self.use_dp = use_dp
        # distributed parallel
        self._init_distributed_vars()
        # init distributed
        self.init_parallel(use_dp=use_dp)
        # running path
        self.running_path_build_config = {
            "running_path": running_path,
            "increment": running_path_increment,
        }
        self.running_path = None
        # model
        self.model = None
        # optimizer
        self.optimizer = None
        # scheduler
        self.scheduler = None
        # device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # train loader
        self.train_loader = None
        # test loader
        self.test_loader = None
        # val loader
        self.val_loader = None
        # config of train dataloader
        self.config_dataloader_train = {}
        # config of test dataloader
        self.config_dataloader_test = {}
        # config of val dataloader
        self.config_dataloader_val = {}

        # gradient scaler for mixed precision training
        self.scaler = (
            torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        )

        # console logger
        self.logger = (
            setup_logger(name=self.logger_name_prefix)
            if not self.is_parallel
            else setup_logger(f"{self.logger_name_prefix}`{self.rank}`")
        )

        # early stop patience
        self.early_stop_patience = early_stop_patience
        # config of user defined
        self.config = config

        # status
        self.current_epoch = 0
        # best metric
        self.best_metric = None
        # early stop counter
        self.early_stop_counter = 0

        # run mode
        self.is_train = False
        # save mode
        self.save_freq = False

    @property
    def tb_logger(self):
        if self._tb_logger is None:
            self.setup_tb_logger()
        return self._tb_logger

    @tb_logger.setter
    def tb_logger(self, value):
        self._tb_logger = value

    @staticmethod
    def sent_broadcast(content):
        import torch.distributed as dist
        object_list = [content]
        dist.broadcast_object_list(object_list, src=0)

    @staticmethod
    def receive_broadcast():
        import torch.distributed as dist
        object_list = [{}]
        dist.broadcast_object_list(object_list, src=0)
        return object_list[0]

    @staticmethod
    def sync_loss(loss, op: torch.distributed.ReduceOp.RedOpType = torch.distributed.ReduceOp.AVG):
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss, op=op)

    def _setup_coordinated_base_path(self, running_path, increment):
        """
        coordinate base path creation across all processes | 协调所有进程的基础路径创建
        This ensures consistent path usage regardless of training mode | 这确保了无论训练模式如何都有一致的路径使用
        Args:
            running_path: base path template | 基础路径模板
            increment: whether to increment path if exists | 如果存在是否递增路径
        Returns:
            str: master-coordinated base running path | 主进程协调的基础运行路径
        """

        if self.is_parallel:
            # unified distributed approach for both single-machine and multi-machine | 为单机和多机统一的分布式方法
            if self._is_master():
                # master process exclusively handles base path creation | 主进程专门处理基础路径创建
                base_running_path = increment_path(
                    running_path, sep="_", mkdir=False, increment=increment
                )
                # broadcast the finalized path to all processes | 向所有进程广播最终路径
                self.sent_broadcast(base_running_path)
            else:
                # all non-master processes receive the master-created path | 所有非主进程接收主进程创建的路径
                base_running_path = self.receive_broadcast()
        else:
            # non-distributed scenarios | 非分布式场景
            base_running_path = increment_path(
                running_path, sep="_", mkdir=False, increment=increment
            )

        if self.is_parallel and not self.use_dp:
            self.running_path = increment_path(
                base_running_path / f"rank_{self.rank}",
                sep="_",
                mkdir=True,
                increment=False,
            )
        else:
            self.running_path = increment_path(
                base_running_path, sep="_", mkdir=True, increment=False
            )

    def _init_distributed_vars(self):
        import os

        self.cuda_available = torch.cuda.is_available()
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", -1))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.is_parallel = self.world_size > 1

    def _is_master(self):
        return self.rank == -1 or self.rank == 0

    def _setup_cpu_mode(self):
        """
        set the CPU operating mode | 设置CPU运行模式
        """
        self.is_parallel = False
        self.rank = -1
        self.local_rank = -1
        self.device = torch.device("cpu")
        self.model.to(self.device)

    def _setup_single_machine_single_gpu_mode(self):
        """
        set the single-machine-single-gpu operating mode | 设置单机单卡运行模式
        """
        self.is_parallel = False
        self.rank = -1
        self.local_rank = 0
        self.device = torch.device("cuda:0")
        self.model.to(self.device)

    def _setup_single_machine_multi_gpu_mode(
            self,
            device_ids: Optional[list] = None
    ):
        """
        set the single-machine-multi-gpu operating mode | 设置单机多卡运行模式
        Args:
            device_ids: list of device ids for parallel training | 并行训练的设备ID列表
        """
        import os

        self.is_parallel = True
        required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK"]
        missing_vars = [var for var in required_vars if var not in os.environ]
        if missing_vars:
            self.logger.info(
                f"""
                Distributed environment variables missing:{missing_vars}
                分布式环境变量缺失: {missing_vars}

                Parallel will be performed using DataParallel 
                将使用`DataParallel`执行并行

                Please make sure to start your script by using `torchrun --nproc_per_node= <number of gpus>` or `python -m torch.distributed.launch --nproc_per_node= <number of gpus>`
                请确保通过使用 x 或 y 启动您的脚本
                """
            )
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.model.to(self.device)
        if missing_vars or self.use_dp:
            self.model = torch.nn.DataParallel(
                self.model,
                device_ids=device_ids
            )
            return False
        else:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
            return True

    def _setup_multi_machine_multi_gpu_mode(self):
        """
        set the multi-machine-multi-gpu operating mode | 设置多机多卡运行模式
        """
        self.is_parallel = True
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.model.to(self.device)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=False,
        )

    def _validate_epoch(self, model=None, **kwargs) -> Dict[str, any]:
        """
        validate one epoch | 验证一个 epoch

        Returns:
            metrics: dict of metrics; the evaluate() method return | 指标字典； evaluate()方法返回的指标字典

        """
        if self.is_parallel and not self._is_master():
            # for non-master processes, return empty metrics | 对于非主进程，返回空指标
            return {}

        self.model.eval()
        with torch.no_grad():
            metrics = self.evaluate(model=model, **kwargs)
        return metrics

    def load_tb_logger(self):
        self._setup_coordinated_base_path(**self.running_path_build_config)
        # tensorboard logger
        self.tb_logger = SummaryWriter(log_dir=str(Path(self.running_path) / "tb_log"))

    def get_parallel_sampler(self, dataset, shuffle=True):
        """
        get parallel sampler | 获取并行采样器
        Args:
            dataset: dataset | 数据集
            shuffle: shuffle | 是否打乱数据

        Returns:
            sampler(torch.utils.data.Sampler): parallel sampler | 并行采样器
        """
        if self.is_parallel and not self.use_dp and torch.distributed.is_initialized():
            from torch.utils.data.distributed import DistributedSampler

            return DistributedSampler(
                dataset, shuffle=shuffle, rank=self.rank, num_replicas=self.world_size
            )
        else:
            return None

    def set_config_dataloader(self, mode, **kwargs):
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode must be train/val/test, but got {mode}")

        config_dataloader = {}
        for k, v in kwargs.items():
            config_dataloader[k] = v

        if mode == "train":
            self.config_dataloader_train = config_dataloader
        elif mode == "test":
            self.config_dataloader_test = config_dataloader
        else:
            self.config_dataloader_val = config_dataloader

    def load_dataloader(self, mode="train", **kwargs) -> torch.utils.data.DataLoader:
        """
        load dataloader | 加载数据加载器
        Args:
            mode(str): mode of dataloader | 数据加载器模式 (train/val/test)
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError(f"mode must be train/val/test, but got {mode}")

        if mode == "train":
            dataset = self.load_dataset_train()
            sampler = self.get_parallel_sampler(dataset, shuffle=True)
            if sampler is not None:
                dataloader_kwargs = self.config_dataloader_train.copy()
                dataloader_kwargs["shuffle"] = False
                dataloader_kwargs["sampler"] = sampler
                return DataLoader(dataset, **dataloader_kwargs)
            else:
                return DataLoader(dataset, **self.config_dataloader_train)
        elif mode == "test":
            dataset = self.load_dataset_test()
            return DataLoader(dataset, **self.config_dataloader_test)
        else:
            dataset = self.load_dataset_val()
            return DataLoader(dataset, **self.config_dataloader_val)

    def load_optimizer(self, **kwargs):
        """
        load optimizer | 加载优化器
        """
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def load_scheduler(self, **kwargs):
        """
        load scheduler | 加载学习率调度器
        """
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def save_checkpoint(self, name="model.pt", path=None, **kwargs):
        """
        save checkpoint | 保存模型检查点
        Args:
            name(str): name of checkpoint file | 检查点文件名
            path(str): path to save checkpoint file | 检查点文件保存路径
        """
        checkpoint = {
            "epoch": self.current_epoch + 1,
            "model": (
                self.model.module
                if isinstance(
                    self.model,
                    (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
                )
                else self.model
            ),
            "model_state_dict": (
                self.model.module.state_dict()
                if isinstance(
                    self.model,
                    (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
                )
                else self.model.state_dict()
            ),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.config.__dict__ if hasattr(self.config, "__dict__") else {},
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        }
        if path is None:
            path = self.running_path
            Path(path).mkdir(parents=True, exist_ok=True)
            filepath = Path(path) / name
        else:
            filepath = Path(path) / name
        torch.save(checkpoint, filepath)
        if self.logger is not None:
            self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(
        self,
        filepath: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        **kwargs,
    ):
        """
        load checkpoint | 加载模型检查点

        Args:
            filepath: path to checkpoint file | 检查点文件路径
            load_optimizer(bool): whether to load optimizer | 是否加载优化器
            load_scheduler(bool): whether to load scheduler | 是否加载学习率调度器
        """
        from packaging import version

        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            checkpoint = torch.load(
                filepath, map_location=self.device, weights_only=False
            )
        else:
            checkpoint = torch.load(filepath, map_location=self.device)
        if self.model is None:
            raise ValueError(
                "model must be loaded before loading checkpoint | 加载检查点前必须加载模型"
            )

        # load model state dict
        if isinstance(
            self.model,
            (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel),
        ):
            self.model.module.load_state_dict(checkpoint.get("model_state_dict"))
        else:
            self.model.load_state_dict(checkpoint.get("model_state_dict"))

        # load optimizer state dict
        if load_optimizer and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))

        # load scheduler state dict
        if (
            load_scheduler
            and checkpoint.get("scheduler_state_dict") is not None
            and self.scheduler is not None
        ):
            self.scheduler.load_state_dict(checkpoint.get("scheduler_state_dict"))

        self.current_epoch = checkpoint.get("epoch", 1) - 1

        if self.logger is not None:
            self.logger.info(
                f"Checkpoint loaded from {filepath}; the epoch is {self.current_epoch}; save the weight time is {checkpoint.get('timestamp')}"
            )

    def init_parallel(self, use_dp=None, **kwargs):
        """
        init distributed | 初始化并行
        """
        import torch.distributed as dist
        if use_dp is not None:
            self.use_dp = use_dp
        if self.cuda_available:
            if (not self.use_dp) and ((1 < self.device_count == self.world_size) or (self.world_size > self.device_count)):
                if not dist.is_initialized():
                    try:
                        dist.init_process_group(
                            backend=self.parallel_backend,
                            timeout=datetime.timedelta(minutes=30),
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to initialize distributed process group: {e} | 初始化分布式进程组失败: {e}"
                        )

    def setup_parallel(
            self,
            device_ids: Optional[list] = None,
            use_dp: bool = False,
            **kwargs
    ):
        """
        setup parallel training | 设置并行训练
        Args:
            device_ids: list of device ids for parallel training | 并行训练的设备ID列表
            use_dp: use data parallel | 是否使用数据并行
            **kwargs:
        """
        import torch.distributed as dist
        # 模型检查 | Model check
        if self.model is None:
            ValueError("The model is not loaded. Please make sure the model is loaded before setting the training mode [CPU only; single GPU; Single machine with multiple GPUs; Multi-machine and multi-GPU]. | 模型未加载。请确保模型加载后再设置训练模式[仅CPU; 单GPU; 单机多GPU; 多机多GPU]。")
        # 初始化并行 | Initialize parallel
        if dist.is_initialized():
            self.init_parallel()
        self.use_dp = use_dp
        if not self.cuda_available:  # single-machine CPU | 单机CPU
            self._setup_cpu_mode()
            self.logger.info(
                f"""
                The current training mode is: [CPU only]
                当前训练模式为： [仅CPU] 
                """
            )
        elif 1 < self.device_count == self.world_size:  # single-machine multi-GPU | 单机多卡
            flag = self._setup_single_machine_multi_gpu_mode(device_ids)
            self.logger.info(
                f"""
                The current training mode is: [Single machine with multiple GPUs]
                当前训练模式为： [单机多GPU]
    
                device：{self.device}
                运行设备：{self.device}
                
                Parallel mode：{'DDP(DistributedDataParallel)' if flag else 'DP(DataParallel)'}
                并行模式：{'DDP(DistributedDataParallel)' if flag else 'DP(DataParallel)'}
                """
            )
        elif self.world_size > self.device_count:  # multi-machine multi-GPU | 多机多卡
            self._setup_multi_machine_multi_gpu_mode()
            self.logger.info(
                f"""
                The current training mode is: [Multi-machine and multi-GPU]
                当前训练模式为： [多机多GPU]
                
                device：{self.device}
                运行设备：{self.device}
                """
            )
        else:  # single-machine single-GPU | 单机单卡
            self._setup_single_machine_single_gpu_mode()
            self.logger.info(
                f"""
                The current training mode is: [single GPU]
                当前训练模式为： [单GPU]
                
                device：{self.device}
                运行设备：{self.device}
                """
            )

    def is_best_model(self, metrics, **kwargs):
        """
        check if current model is best model | 检查当前模型是否是最佳模型

        * user can reimplement this method to check if current model is best model | 用户可以重新实现此方法来检查当前模型是否是最佳模型
        Args:
            metrics: dict of metrics; the evaluation() method return | 指标字典； evaluation()方法返回的指标字典
        """
        # check if best model
        if self.best_metric["global"] > metrics["global"]:
            return True
        else:
            return False

    def is_early_stop(self):
        return self.early_stop_counter >= self.early_stop_patience

    def update_scheduler(self, metrics=None, **kwargs):
        """
        update scheduler | 更新学习率调度器

        * user can reimplement this method to update scheduler | 用户可以重新实现此方法来更新学习率调度器
        Args:
            metrics: dict of metrics; the evaluate() method return | 指标字典； evaluate()方法返回的指标字典
        """
        self.scheduler.step()

    def run_train(self, epochs=100, save_freq=10, resume=None, weight=None, use_dp=None, **kwargs):
        """
        run train | 运行训练
        Args:
            epochs: number of epochs to train | 训练的 epoch 数
            save_freq: frequency to save checkpoint | 保存检查点的频率
            resume: path to the checkpoint of resume training | 恢复训练的检查点路径
            weight:
            use_dp: whether use DataParallel | 是否使用DataParallel
        """
        # set run mode
        self.is_train = True
        # load model
        self.load_model(**kwargs)
        # init parallel if available
        self.init_parallel(use_dp=use_dp, **kwargs)
        # setup parallel
        self.setup_parallel(use_dp=use_dp)
        # # load tb_logger
        self.load_tb_logger()
        # load dataloader
        self.train_loader = self.load_dataloader(mode="train", **kwargs)
        self.val_loader = self.load_dataloader(mode="val", **kwargs)

        # load optimizer
        self.load_optimizer(**kwargs)
        # load scheduler
        self.load_scheduler(**kwargs)

        # running info
        self.logger.info(f"current running path: {self.running_path}")
        self.logger.info(f"current running device: {self.device}")

        # start from weight
        if weight is not None:
            self.load_checkpoint(weight, load_optimizer=False, load_scheduler=False, **kwargs)
            self.current_epoch = 0
            self.logger.info(f"Start training from weight {weight}")

        # resume training
        if resume is not None:
            self.load_checkpoint(resume, **kwargs)
            self.logger.info(
                f"Resuming training from `{resume}`: epoch {self.current_epoch}"
            )

        # ** start train
        self.logger.info(
            f"Start training from epoch {self.current_epoch + 1} to {epochs}"
        )
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, epochs):
            self.current_epoch = epoch
            # train one epoch
            self.train_epoch(epoch, epochs, **kwargs)

            # validate one epoch and get metrics
            val_metrics = {}
            # broadcast metrics to all processes in distributed mode
            if self.is_parallel:
                import torch.distributed as dist

                if self._is_master():
                    # only master process performs validation
                    if self.val_loader is not None:
                        if isinstance(
                            self.model,
                            (
                                (
                                    torch.nn.DataParallel,
                                    torch.nn.parallel.DistributedDataParallel,
                                )
                            ),
                        ):
                            rawModel = self.model.module
                        else:
                            rawModel = self.model
                        val_metrics = self._validate_epoch(model=rawModel, **kwargs)
                    # broadcast metrics to other processes
                    self.sent_broadcast(val_metrics)
                else:
                    # non-master processes wait and receive metrics
                    val_metrics = self.receive_broadcast()
            else:
                # single process mode, just validate
                val_metrics = self._validate_epoch(self.model)

            # check if best model and save checkpoint - only master process
            early_stop = False
            if self._is_master():
                if not self.save_freq and self.val_loader is not None:
                    # check if best model
                    if self.best_metric is None:
                        self.best_metric = val_metrics
                    else:
                        if self.is_best_model(val_metrics, **kwargs):
                            self.best_metric = val_metrics
                            self.save_checkpoint("best.pt", **kwargs)
                            self.early_stop_counter = 0
                        else:
                            self.early_stop_counter += 1

                    # early stopping
                    if self.is_early_stop():
                        self.logger.info(
                            f"Early stopping triggered after {self.early_stop_patience} epochs without improvement"
                        )
                        early_stop = True
                else:
                    # save checkpoint every `save_freq` epochs when no validation loader
                    if (epoch + 1) % save_freq == 0:
                        self.save_checkpoint(f"model_epoch_{epoch + 1}.pt", **kwargs)

                # save the final model - only master process saves
                self.save_checkpoint("last.pt",**kwargs)
            # update scheduler - all processes should update scheduler
            self.update_scheduler(metrics=val_metrics, **kwargs)

            # broadcast early stop signal to all processes
            if self.is_parallel:
                import torch.distributed as dist

                early_stop_tensor = torch.tensor(
                    [1 if early_stop else 0], device=self.device
                )
                dist.broadcast(early_stop_tensor, src=0)
                early_stop = early_stop_tensor.item() == 1

            # check early stop condition for all processes
            if early_stop:
                break

    def run_eval(self, **kwargs):
        """
        run evaluation | 运行评估
        """
        # set run mode
        self.is_train = False
        if self.model is None:
            raise ValueError(
                "The model is empty. Make sure that self.model is assigned in the overridden load_model() method | 模型为空。请确保重写的load_model()方法中对self.model进行了赋值"
            )
        self.model.eval()
        with torch.no_grad():
            metrics = self.evaluate(self.model, **kwargs)
        self.logger.info("Evaluation completed! | 评估完成!")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                self.logger.info(f"{key}: {value:.4f}")
        else:
            self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def run_infer(self, *args, **kwargs):
        """
        run inference | 运行推理
        """
        # set run mode
        self.is_train = False
        if self.model is None:
            raise ValueError(
                "The model is empty. Make sure that self.model is assigned in the overridden load_model() method | 模型为空。请确保重写的load_model()方法中对self.model进行了赋值"
            )
        self.model.eval()
        with torch.no_grad():
            result = self.inference_step(*args, **kwargs)
        return result

    @abstractmethod
    def load_model(self, *args, **kwargs) -> torch.nn.Module:
        """
        load model | 加载模型

        Returns:
            model(torch.nn.Module): model
        """
        raise NotImplementedError(
            "load_model method must be implemented | 必须实现load_model方法"
        )

    @abstractmethod
    def load_dataset_train(self, **kwargs) -> torch.utils.data.Dataset:
        """
        load train dataset | 加载训练数据集

        Returns:
            train_dataset(torch.utils.data.Dataset): train dataset | 训练数据集
        """
        raise NotImplementedError(
            "load_dataset_train method must be implemented | 必须实现load_dataset_train方法"
        )

    @abstractmethod
    def load_dataset_test(self, **kwargs) -> torch.utils.data.Dataset:
        """
        load test dataset | 加载测试数据集

        Returns:
            test_dataset(torch.utils.data.Dataset): test dataset | 测试数据集
        """
        raise NotImplementedError(
            "load_dataset_test method must be implemented | 必须实现load_dataset_test方法"
        )

    @abstractmethod
    def load_dataset_val(self, **kwargs) -> torch.utils.data.Dataset:
        """
        load val dataset | 加载验证数据集

        Returns:
            val_dataset(torch.utils.data.Dataset): val dataset | 验证数据集
        """
        raise NotImplementedError(
            "load_dataset_val method must be implemented | 必须实现load_dataset_val方法"
        )

    @abstractmethod
    def train_epoch(self, epoch, epochs, **kwargs):
        raise NotImplementedError(
            "train_one_epoch method must be implemented | train_one_epoch方法必须实现"
        )

    @abstractmethod
    def evaluate(self, *args, model=None, **kwargs) -> Dict[str, any]:
        raise NotImplementedError(
            "train_one_epoch method must be implemented | train_one_epoch方法必须实现"
        )

    @abstractmethod
    def inference_step(self, *args, **kwargs):
        """
        run inference step | 运行推理步骤
        Args:
            *args:
            **kwargs:

        Returns:
            any: inference result | 推理结果
        """
        inputs = args[0] if args else kwargs.get("inputs")
        if inputs is not None:
            inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
            return self.model(inputs)
        else:
            raise ValueError(
                "inference must have inputs | inference_step方法必须实现输入数据"
            )
