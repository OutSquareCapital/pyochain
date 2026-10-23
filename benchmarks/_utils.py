from __future__ import annotations

from collections.abc import Callable, Iterable
from enum import IntEnum, StrEnum, auto
from typing import TYPE_CHECKING, Final, Protocol

import pytest

from pyochain import Seq

if TYPE_CHECKING:
    import polars as pl
    from _pytest.mark import ParameterSet

type BenchFn = Callable[[int], object]


SIZES: Final[Seq[int]] = Seq(10, 100, 1_000, 10_000, 100_000, 1_000_000)
SIZE_PARAMS = pytest.mark.parametrize("size", SIZES)


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


class PlEnum(StrEnum):
    """Base class for enums that can be used as polars expressions."""

    def pl(self) -> pl.Expr:
        """Return a polars expression for the enum value."""

        import polars as pl

        return pl.col(self.value)


class Lib(PlEnum):
    """Libraries used in the benchmarks."""

    Pyochain = auto()
    SortedContainers = auto()

    def to_param(self, cls: type) -> ParameterSet:
        return pytest.param(cls, id=f"{self}-{cls.__name__}")


class Sizes(IntEnum):
    SIZE_64 = 64
    SIZE_256 = 256
    SIZE_1024 = 1024
    SIZE_4096 = 4096


class VariantGroups(StrEnum):
    CREATE = auto()
    MAP = auto()
    AND_THEN = auto()
    MATCH = auto()


def identity[T](x: T) -> T:
    return x


def identity_vargs[T](*args: T) -> tuple[T, ...]:
    return args


def consume(values: Iterable[object]) -> None:
    for _ in values:
        pass
