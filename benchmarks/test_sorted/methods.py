"""Methods dispatchs to handle API divergences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
import sortedcontainers as sc

from pyochain import Iter
from pyochain.collections import SortedList

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from _pytest.mark import ParameterSet

    from pyochain.abc import PyoSequence

    from ._utils import Set, SortedCollection


@dataclass(slots=True)
class BaseOp:
    name: str

    @staticmethod
    def generate(*ops: BaseOp) -> Sequence[ParameterSet]:
        return Iter(ops).map(lambda slf: pytest.param(slf, id=slf.name)).collect(tuple)


@dataclass(slots=True)
class SetOp(BaseOp):
    fn: Callable[[Set, Iterable[int]], object]
    halved: bool = True
    mut: bool = False


@dataclass(slots=True)
class OpFactory[T](BaseOp):
    fn: Callable[[T], Callable[[int], object]]


def update(
    cls: type[SortedCollection],
) -> Callable[[SortedCollection, Iterable[int]], None]:
    if cls is sc.SortedList:
        return sc.SortedList[int].update  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].extend  # pyright: ignore[reportReturnType]


def add(cls: type[SortedCollection]) -> Callable[[SortedCollection, int], None]:
    if cls is sc.SortedList:
        return sc.SortedList[int].add  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].append  # pyright: ignore[reportReturnType]


def delitem(values: PyoSequence[int]) -> Callable[[SortedCollection], None]:
    return lambda x: values.iter().for_each(x.__delitem__)
