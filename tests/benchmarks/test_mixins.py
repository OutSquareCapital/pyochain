from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Self

import pytest

from pyochain.abc import Fluent

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._utils import BenchFixture


@dataclass(slots=True)
class FooRust(Fluent):
    value: int


@dataclass(slots=True)
class FooPy:
    value: int

    def pipe[**P, R](
        self, fn: Callable[Concatenate[Self, P], R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        return fn(self, *args, **kwargs)


type Foo = FooRust | FooPy


def _without_args_or_kwargs(_foo: Foo) -> int:
    return 1


def _with_one_arg(_foo: Foo, _arg1: int) -> int:
    return 1


def _with_two_args(_foo: Foo, _arg1: int, _arg2: int) -> int:
    return 1


def _with_one_kwarg(_foo: Foo, *, _kwarg1: int) -> int:
    return 1


def _with_three_kwargs(_foo: Foo, *, _kwarg1: int, _kwarg2: int, _kwarg3: int) -> int:
    return 1


def _with_args_and_kwargs(
    _foo: Foo, _arg1: int, _arg2: int, *, _kwarg1: int, _kwarg2: int
) -> int:
    return 1


compared = pytest.mark.parametrize(
    "foo",
    [
        pytest.param(FooRust(10), id="native"),
        pytest.param(FooPy(10), id="python"),
    ],
)


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_without_args_or_kwargs(benchmark: BenchFixture, foo: Foo) -> None:
    # pyrefly: ignore[bad-argument-type]
    assert benchmark(foo.pipe, _without_args_or_kwargs) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_with_one_arg(benchmark: BenchFixture, foo: Foo) -> None:
    # pyrefly: ignore[bad-argument-type]
    assert benchmark(foo.pipe, _with_one_arg, 3) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_with_two_args(benchmark: BenchFixture, foo: Foo) -> None:
    # pyrefly: ignore[bad-argument-type]
    assert benchmark(foo.pipe, _with_two_args, 3, 5) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_with_one_kwarg(benchmark: BenchFixture, foo: Foo) -> None:
    # pyrefly: ignore[bad-argument-type]
    assert benchmark(foo.pipe, _with_one_kwarg, _kwarg1=3) == 1  # pyright: ignore[reportArgumentType]


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_with_three_kwargs(benchmark: BenchFixture, foo: Foo) -> None:
    assert (
        benchmark(
            # pyrefly: ignore[bad-argument-type]
            foo.pipe,  # pyright: ignore[reportArgumentType]
            _with_three_kwargs,
            _kwarg1=3,
            _kwarg2=5,
            _kwarg3=7,
        )
        == 1
    )


@pytest.mark.benchmark(group="mixin_into")
@compared
def test_pipe_with_args_and_kwargs(benchmark: BenchFixture, foo: Foo) -> None:
    assert (
        benchmark(
            # pyrefly: ignore[bad-argument-type]
            foo.pipe,  # pyright: ignore[reportArgumentType]
            _with_args_and_kwargs,
            3,
            5,
            _kwarg1=7,
            _kwarg2=11,
        )
        == 1
    )
