from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol

import pytest
from sortedcontainers import SortedList as SortedListPy

from pyochain import Range
from pyochain.collections import SortedList

from ._utils import List, assert_stop_iter, stop_iter_or_unsupported, update_list

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from _pytest.mark import ParameterSet


def _method_param[T](cls: type[T], f: Callable[[T], object]) -> ParameterSet:
    return pytest.param(cls, f, id=f"{cls.__module__}.{cls.__name__}.{f.__name__}")


def _list_slice(
    lst: list[int], start: int, stop: int, *, reverse: bool = False
) -> Iterator[int]:
    it = iter(lst) if not reverse else reversed(lst)
    return itertools.islice(it, start, stop)


REVERSE_PARAM = pytest.mark.parametrize("reverse", (False, True))
BOUNDED_PARAMS = pytest.mark.parametrize(
    ("cls", "f"),
    ((
        _method_param(SortedListPy, SortedListPy[int].irange),
        _method_param(SortedListPy, SortedListPy[int].islice),
        _method_param(SortedList, SortedList[int].irange),
        _method_param(SortedList, SortedList[int].islice),
        pytest.param(list, _list_slice),
    )),
)


class SliceFn[T](Protocol):
    def __call__(
        self,
        it: T,
        start: int,
        stop: int,
        reverse: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
    ) -> Iterator[object]: ...


@REVERSE_PARAM
@BOUNDED_PARAMS
def test_clear[T: List](cls: type[T], f: SliceFn[T], *, reverse: bool) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: 10**9 + i))
    it = f(sl, 0, 10**9 + length, reverse=reverse)
    _ = next(it)
    sl.clear()
    stop_iter_or_unsupported(sl, it)


@REVERSE_PARAM
@BOUNDED_PARAMS
def test_clear_then_update[T: List](
    cls: type[T], f: SliceFn[T], *, reverse: bool
) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: 10**9 + i))
    it = f(sl, 0, 10**9 + length, reverse=reverse)
    _ = next(it)
    sl.clear()
    update_list(sl, [10**9 + length])
    stop_iter_or_unsupported(sl, it)


@REVERSE_PARAM
@BOUNDED_PARAMS
def test_pop[T: List](cls: type[T], f: SliceFn[T], *, reverse: bool) -> None:
    length = 10
    sl = cls(Range(length).iter().map(lambda i: 10**9 + i))
    it = f(sl, 0, 10**9 + length, reverse=reverse)
    _ = next(it)
    for _ in range(length - 1):
        _ = sl.pop()
    match sl:
        case SortedListPy():
            # sortedcontainers will try to access a non-existing index, which raises IndexError instead of StopIteration.
            with pytest.raises(IndexError):
                _ = next(it)
        case _:
            assert_stop_iter(it)
