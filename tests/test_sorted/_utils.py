from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

from sortedcontainers import SortedList as SortedListPy

from pyochain.collections import SortedList

type IntoIter[T] = Callable[[T], Iterator[object]]
type List = list[int] | SortedList[int] | SortedListPy[int]


def update_list(sl: List, values: Iterable[int]) -> None:
    match sl:
        case list() | SortedList():
            sl.extend(values)
        case _:
            sl.update(values)
