from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Protocol

import pytest
from sortedcontainers import SortedList as PySortedList

from pyochain import Iter, Range
from pyochain.collections import SortedList

from ._utils import (
    LOAD,
    List,
    assert_stop_iter,
    method_param,
    stop_iter_or_unsupported,
    update_list,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def _list_slice(
    lst: list[int], start: int, stop: int, *, reverse: bool = False
) -> Iterator[int]:
    it = iter(lst) if not reverse else reversed(lst)
    bounds = (
        (start, stop) if not reverse else (max(len(lst) - stop, 0), len(lst) - start)
    )
    return itertools.islice(it, *bounds)


REVERSE_PARAM = pytest.mark.parametrize("reverse", (False, True))
BOUNDED_PARAMS = pytest.mark.parametrize(
    ("cls", "f"),
    ((
        method_param(PySortedList, PySortedList[int].irange),
        method_param(PySortedList, PySortedList[int].islice),
        method_param(SortedList, SortedList[int].irange),
        method_param(SortedList, SortedList[int].islice),
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
        case PySortedList():
            # sortedcontainers will try to access a non-existing index, which raises IndexError instead of StopIteration.
            with pytest.raises(IndexError):
                _ = next(it)
        case _:
            assert_stop_iter(it)


@BOUNDED_PARAMS
def test_remove_below_lower_bound[T: List](cls: type[T], f: SliceFn[T]) -> None:
    remove_range = 100
    stop = LOAD * 3
    sl = cls(range(stop))
    it = f(sl, LOAD - remove_range, stop, reverse=True)
    for value in range(remove_range):
        sl.remove(value)
    match sl:
        case list():
            assert_stop_iter(it)
        case PySortedList():
            _iter_after_removed_lower_bound(it, IndexError)
        case SortedList():
            _iter_after_removed_lower_bound(it, StopIteration)


def _iter_after_removed_lower_bound(it: Iterator[object], err: type[Exception]) -> None:
    iterator = Iter(it).skip(LOAD * 2 - 1)
    assert iterator.next().is_some()
    with pytest.raises(err):
        _ = next(iterator)
