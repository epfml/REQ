from .nero import Nero
from .nerom import NeroM
from .rotational_adamw import RotationalAdamW
from .rotational_sgd import RotationalSGD
from .rotational_lion import RotationalLion

optimizer_factories = dict(
    nero=Nero,
    nerom=NeroM,
    rvsgd=RotationalSGD,
    rvadamw=RotationalAdamW,
    rvlion=RotationalLion,
)
