from .rotational_adamw import RotationalAdamW
from .rotational_sgd import RotationalSGD
from .rotational_lion import RotationalLion
from .rotational_wrapper import RotationalWrapper
from .rotational_speed_controlled_adamw import RotationalSpeedControlledAdamW
from .rotational_frozen_adamw import BiasFrozenAdamW

optimizer_factories = dict(
    rvsgd=RotationalSGD,
    rvadamw=RotationalAdamW,
    rvlion=RotationalLion,
    rvwrapper=RotationalWrapper,
    rvcadamw=RotationalSpeedControlledAdamW,
    fadamw=BiasFrozenAdamW
)
