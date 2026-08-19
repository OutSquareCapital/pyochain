from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from typing import Protocol

from rich.traceback import install

from pyochain import Dict

_ = install(show_locals=True)

type BenchFn = Callable[[int], object]


class BenchFixture(Protocol):
    def __call__[**P, T](
        self,
        function_to_benchmark: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...


class Sizes(IntEnum):
    SIZE_64 = 64
    SIZE_256 = 256
    SIZE_1024 = 1024
    SIZE_4096 = 4096


SIZES = Dict({10: 10_000, 100: 1000, 1_000: 100, 10_000: 10})


class VariantGroups(StrEnum):
    CREATE = auto()
    MAP = auto()
    AND_THEN = auto()
    MATCH = auto()
