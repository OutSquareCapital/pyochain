from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from typing import Final, Protocol

import pytest

from pyochain import Seq

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


SIZES: Final[Seq[int]] = Seq(10, 100, 1_000, 10_000, 100_000)


class VariantGroups(StrEnum):
    CREATE = auto()
    MAP = auto()
    AND_THEN = auto()
    MATCH = auto()


def identity[T](x: T) -> T:
    return x


def identity_vargs[T](*args: T) -> tuple[T, ...]:
    return args


WINDOW_CASES: Final = pytest.mark.parametrize(
    ("size", "tup_len"),
    SIZES
    .iter()
    .product(SIZES.iter().map(lambda x: x // 10))
    .filter_star(lambda size, tup_len: tup_len < size)
    .collect(tuple),
)
