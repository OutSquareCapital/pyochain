import random
from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from functools import cache
from math import sqrt
from typing import Final, Protocol

import pytest

from pyochain import Seq, Vec

type BenchFn = Callable[[int], object]


class BenchFixture(Protocol):
    def __call__[**P, T](
        self,
        function_to_benchmark: Callable[P, T],
        /,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> T: ...

    def pedantic[*Ts, T](  # ruff: ignore[too-many-arguments, too-many-positional-arguments]
        self,
        target: Callable[[*Ts], T],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
        setup: Callable[[], tuple[tuple[*Ts], dict[str, object]]] | None = None,
        teardown: Callable[[], None] | None = None,
        rounds: int = 1,
        warmup_rounds: int = 0,
        iterations: int = 1,
    ) -> T: ...


class Sizes(IntEnum):
    SIZE_64 = 64
    SIZE_256 = 256
    SIZE_1024 = 1024
    SIZE_4096 = 4096


SIZES: Final[Seq[int]] = Seq(100, 1_000, 10_000, 100_000, 1_000_000)


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
    .product((2, 4, 8, 16, 32, 64, 128))
    .filter_star(lambda size, tup_len: tup_len < size)
    .collect(tuple),
)


@cache
def shuffled(size: int) -> Vec[int]:
    return Vec.wrap(random.Random(0).sample(range(size), size))


def rounds_for(size: int) -> tuple[int, int]:
    rounds = max(20, int(50 * sqrt(SIZES.last() // size)))
    warmup = rounds // 20
    return rounds, warmup
