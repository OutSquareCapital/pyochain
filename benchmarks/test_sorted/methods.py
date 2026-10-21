"""Methods dispatchs to handle API divergences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedList as PySortedList

from pyochain import Iter
from pyochain.collections import SortedList

from ._utils import Dict, List, Set

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from _pytest.mark import ParameterSet

    from pyochain.abc import PyoSequence


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


def update[L: List](cls: type[L]) -> Callable[[L, Iterable[int]], None]:
    if cls is PySortedList:
        return PySortedList[int].update  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].extend  # pyright: ignore[reportReturnType]


def add[L: List](cls: type[L]) -> Callable[[L, int], None]:
    if cls is PySortedList:
        return PySortedList[int].add  # pyright: ignore[reportReturnType]
    else:
        return SortedList[int].append  # pyright: ignore[reportReturnType]


def delitem[T: Dict | List](values: PyoSequence[int]) -> Callable[[T], None]:
    return lambda x: values.iter().for_each(x.__delitem__)
