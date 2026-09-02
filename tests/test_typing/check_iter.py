from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Never, assert_never, assert_type

from pyochain import Iter, Range, Seq
from pyochain.abc import PyoIterator

from ._utils import Animal, Dog

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def check_iter_covariance() -> None:
    a = assert_type(Iter(Dog()), Iter[Dog])
    b = assert_type(a.batched(1).flatten(), PyoIterator[Dog])
    _a = assert_type(a.batched(1).map_star(str), PyoIterator[str])
    _b: Iter[Animal] = a
    _c: PyoIterator[Animal] = a
    _f: Iterator[Animal] = a
    _g: PyoIterator[Animal] = b
    _j: Iterator[Animal] = b


def check_iter_flatten() -> Never:
    nested = (
        Range(3)
        .iter()
        .map(
            lambda x: (
                Range(x)
                .iter()
                .map(lambda y: Range(y).iter().map(lambda z: Range(z).pipe(list)))
            )
        )
    )
    _ = assert_type(nested, PyoIterator[PyoIterator[PyoIterator[list[int]]]])
    one = assert_type(nested.flatten(), PyoIterator[PyoIterator[list[int]]])
    two = assert_type(one.flatten(), PyoIterator[list[int]])
    ok = assert_type(two.flatten(), PyoIterator[int])
    # Expected to fail
    _fail = assert_type(ok.flatten(), Never)


def check_chain_covariance[T, S](base: Iterable[T], *others: Iterable[S]) -> None:
    _ = assert_type(Iter(base).chain(*others), PyoIterator[T | S])
    _ = assert_type(itertools.chain(base, *others), itertools.chain[T | S])


def check_map_juxt() -> None:
    funcs = (float, str, bool)
    out = (
        Range(3)
        .iter()
        .map_juxt(float, str, bool, lambda x: [x, "hello"], *funcs)
        .filter_star(
            lambda f, s, b, lst, _, _a, _b: (
                f > 1.0 and s == "2" and b is True and lst[1] == "hello"
            )
        )
        .collect(Seq)
    )
    _ = assert_type(
        out, Seq[tuple[float, str, bool, list[int | str], float, str, bool]]
    )


def check_map_star() -> Never:
    out = (
        Range(3)
        .iter()
        .map(lambda x: (x, str(x), bool(x)))
        .map_star(lambda x, s, b: (x + 1, s + "!", not b))
        .collect(Seq)
    )
    _ = assert_type(out, Seq[tuple[int, str, bool]])
    # Expected to fail
    _ = assert_never(Range(3).iter().map_star(str))


type TupItem = tuple[int, str, bool]


def check_map_windows() -> None:
    def foo(x: int, _y: int) -> int:
        return x

    def bar(*x: int) -> int:
        return sum(x)

    def baz(x: tuple[int, ...]) -> int:
        return sum(x)

    data = Range(3)
    _ = assert_type(data.iter().map_windows_star(1, foo), Any)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]
    _ = assert_type(data.iter().map_windows_star(2, foo), PyoIterator[int])
    _ = assert_type(data.iter().map_windows(3, baz), PyoIterator[int])
    _ = assert_type(data.iter().map_windows_star(2, bar), PyoIterator[int])
    _ = assert_type(data.iter().map_windows_star(3, bar), PyoIterator[int])
