import os
from typing import TypeAlias, TypedDict, Union

from matplotlib.typing import ColorType

StrPath = Union[os.PathLike[str], str]

Quadruple: TypeAlias = tuple[float, float, float, float]


class RectangleKwargs(TypedDict, total=False):
    linewidth: float
    edgecolor: ColorType
    facecolor: ColorType
    alpha: float
