from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from typing import Final, Protocol

from pyochain import Dict

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


SIZES: Final[Dict[int, range]] = Dict({
    10: range(10_000),
    100: range(1000),
    1_000: range(100),
    10_000: range(10),
})


class VariantGroups(StrEnum):
    CREATE = auto()
    MAP = auto()
    AND_THEN = auto()
    MATCH = auto()
