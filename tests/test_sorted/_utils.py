from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Protocol

import pytest
from sortedcontainers import SortedDict as SortedDictPy
from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedSet as SortedSetPy

from pyochain.collections import SortedDict, SortedList, SortedSet

if TYPE_CHECKING:
    from _pytest.mark import MarkDecorator
    from _pytest.mark.structures import ParameterSet

type IntoIter[T] = Callable[[T], Iterator[object]]
type List = list[int] | SortedList[int] | PySortedList[int]
type AnySet = set[int] | SortedSet[int] | SortedSetPy[int]
type Dict = dict[int, object] | SortedDict[int, object] | SortedDictPy[int, object]
type IterStop = type[StopIteration | IndexError]


class SliceFn[T](Protocol):
    def __call__(
        self,
        it: T,
        start: int,
        stop: int,
        reverse: bool = False,  # ruff: ignore[boolean-default-value-positional-argument]
    ) -> Iterator[object]: ...


LOAD = 1000
"""Actuall load size, correspond to desired sublist size."""


def method_param[T](cls: type[T], f: Callable[[T], object]) -> ParameterSet:
    return pytest.param(cls, f, id=f"{cls.__module__}.{cls.__name__}.{f.__name__}")


def bool_param(name: str) -> MarkDecorator:
    return pytest.mark.parametrize(name, (True, False))


def list_slice(
    lst: list[int], start: int, stop: int, *, reverse: bool = False
) -> Iterator[int]:
    it = iter(lst) if not reverse else reversed(lst)
    bounds = (
        (start, stop) if not reverse else (max(len(lst) - stop, 0), len(lst) - start)
    )
    return itertools.islice(it, *bounds)


INTO_ITER_PARAMS = pytest.mark.parametrize("into_iter", (iter, reversed))
UPDATE_PARAMS = bool_param("update")
REVERSE_PARAM = bool_param("reverse")
BOUNDED_PARAMS = pytest.mark.parametrize(
    ("cls", "f"),
    ((
        method_param(PySortedList, PySortedList[int].irange),
        method_param(PySortedList, PySortedList[int].islice),
        method_param(SortedList, SortedList[int].irange),
        method_param(SortedList, SortedList[int].islice),
        pytest.param(list, list_slice),
    )),
)

LIST_CLASSES = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(PySortedList, id="sortedcontainers"),
        pytest.param(list),
        pytest.param(SortedList, id="pyochain"),
    ),
)


def assert_stop_iter(it: Iterator[object], err: IterStop = StopIteration) -> None:
    with pytest.raises(err):
        _ = next(it)


def update_list(sl: List, values: Iterable[int]) -> None:
    match sl:
        case list() | SortedList():
            sl.extend(values)
        case _:
            sl.update(values)
