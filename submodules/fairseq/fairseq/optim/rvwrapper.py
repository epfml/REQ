""" AdamW Optimizer
Impl copied from PyTorch master

NOTE: Builtin optim.AdamW is used by the factory, this impl only serves as a Python based reference, will be removed
someday
"""
import math
import torch
from torch.optim.optimizer import Optimizer
from . import LegacyFairseqOptimizer, register_optimizer
from shared.modules.factories import get_function_dict
from fairseq.dataclass import FairseqDataclass
from fairseq.optim import FairseqOptimizer, register_optimizer
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
from omegaconf import II, OmegaConf
from shared.optimizers.rotational_wrapper import RotationalWrapper
import json


@dataclass
class FairseqRVWrapperConfig(FairseqDataclass):
    weight_decay: float = field(default=0.0, metadata={"help": "weight decay"})
    use_old_adam: bool = field(
        default=False, metadata={"help": "Use fairseq.optim.adam.Adam"}
    )
    opt_functional_cfg: Optional[str] = field(
        default="torch",
        metadata={"help": "Custom functions used for the optimizer computation"}
    )
    lr: List[float] = II("optimization.lr")
    wrapper_inner_type: Optional[str] = field(
        default="adam",
        metadata={"help": "Inner optimizer for gradient direction"}
    )
    wrapper_etar_func: Optional[str] = field(
        default="adamw",
        metadata={"help": "Outer optimizer for update size"}
    )
    wrapper_hyper_parameters: str = field(
        default="{}",
        metadata={"help": "Additional hyper-parameter"}
    )

@register_optimizer("rvwrapper", dataclass=FairseqRVWrapperConfig)
class FairseqRVWrapper(FairseqOptimizer):
    r"""Implements AdamW algorithm.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, cfg: FairseqRVWrapperConfig, params):
        super().__init__(cfg)
        defaults = dict(
            update_norm_decay_factor=0.99,
            per_feature=True,
            zero_mean=True,
            rotational_eps=1e-8,
            zero_mean_on_init=True,
            **self.optimizer_config,
        )
        self._optimizer = FairseqInnerRVWrapper(params, **defaults)

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            "lr": self.cfg.lr[0],
            "weight_decay": self.cfg.weight_decay,
            "inner_type": self.cfg.wrapper_inner_type,
            "etar_func": self.cfg.wrapper_etar_func,
            **json.loads(self.cfg.wrapper_hyper_parameters)
        }
    
class FairseqInnerRVWrapper(Optimizer):
    def __init__(
        self,
        params,
        **defaults
    ):
        self._optimizer = RotationalWrapper(params, **defaults)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._optimizer.step()
        return loss