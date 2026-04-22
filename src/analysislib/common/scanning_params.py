from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Optional
from typing_extensions import assert_never


@dataclass(frozen=True)
class ScanningParameter:
    name: str
    unit: str
    friendly_name: Optional[str] = None

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_h5_tuple(cls, tup) -> ScanningParameter:
        name: bytes
        units: bytes
        expr: bytes
        name, units, expr = tup
        return cls(name.decode('utf-8'), units.decode('utf-8'))

    def axis_label(self, unit: Optional[str] = None):
        namestr = self.friendly_name if self.friendly_name is not None else self.name
        if unit is None:
            plot_unit = self.unit
        else:
            plot_unit = unit
        unitstr = f' ({plot_unit})' if plot_unit != '' else ''
        return f'{namestr}{unitstr}'


class ScanningParameters:
    params: tuple[ScanningParameter, ...]
    param_inds: dict[str, int]  # maybe just store dict[str, ScanningParameter]?

    def __init__(self, params: Sequence[ScanningParameter]):
        self.params = tuple(params)
        self.param_inds = {
            param.name: i
            for i, param in enumerate(self.params)
        }

    def __getitem__(self, key: int | str) -> ScanningParameter:
        if isinstance(key, int):
            index = key
            return self.params[index]
        elif isinstance(key, str):
            name = key
            return self.params[self.param_inds[name]]
        else:
            assert_never(key)

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self):
        return iter(self.params)

    @classmethod
    def from_h5_tuples(cls, iterable: Iterable) -> ScanningParameters:
        return cls([ScanningParameter.from_h5_tuple(tup) for tup in iterable])
