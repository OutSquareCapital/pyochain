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
    def pedantic(  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        target: Callable[..., object],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        setup: Callable[[], tuple[tuple[object, ...], dict[str, object]]] | None = None,
        teardown: Callable[[], None] | None = None,
        rounds: int = 1,
        warmup_rounds: int = 0,
        iterations: int = 1,
    ) -> object: ...


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
