import warnings

import torch

from sirarm_template.utils.ops import parse_version


def get_grad_scaler():
    torch_version = parse_version(torch.__version__)
    try:
        if torch_version >= (2, 1):
            from torch.amp import GradScaler
            scaler = GradScaler("cuda")
        else:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        return scaler
    except (ImportError, AttributeError):
        warnings.warn(
            "使用兼容模式加载 GradScaler (torch.cuda.amp)，建议检查 PyTorch 版本",
            UserWarning
        )
        from torch.cuda.amp import GradScaler
        return GradScaler() if torch.cuda.is_available() else None


def load_checkpoint_support_submodule(
        model: torch.nn.Module,
        checkpoint: str,
        param_name: str = "model",
        strict=True,
        sub_name: list | None = None,
        load_args: dict = {}
):
    if parse_version(torch.__version__) >= (2, 6, 0):
        load_args = {
            **load_args,
            "weights_only": False
        }

    if sub_name is None:
        if isinstance(checkpoint, str):
            checkpoint = torch.load(checkpoint, **load_args)
        model.load_state_dict(checkpoint[param_name], strict=strict)
    else:
        if len(sub_name) > 1:
            sub_model = getattr(model, sub_name[0])
            load_checkpoint_support_submodule(sub_model, checkpoint, param_name, strict, sub_name[1:], load_args)
        elif len(sub_name) == 1:
            sub_model = getattr(model, sub_name[0])
            load_checkpoint_support_submodule(sub_model, checkpoint, param_name, strict, None, load_args)
        else:
            raise ValueError("sub_name must be a list")
    return model


def is_parallel_model( model):
    if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return True
    else:
        return False
