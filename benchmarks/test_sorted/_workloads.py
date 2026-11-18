from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from pyochain import Iter, Vec

from .._utils import consume
from ._utils import RANDOM, AddFn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from _pytest.mark import ParameterSet

    from ._utils import List

type WorkLoad = Callable[[List, int, int, AddFn], object]

WORKLOADS: Vec[ParameterSet] = Vec()
"""List of workloads to benchmark."""


def _register(fn: WorkLoad) -> WorkLoad:
    WORKLOADS.append(pytest.param(fn, id=fn.__name__.removeprefix("_")))
    return fn


@_register
def _priority_queue(sl: List, value: int, pos: int, add_fn: AddFn) -> None:
    if pos < 40:
        add_fn(sl, value)
    elif pos < 80:
        _ = sl.pop()
    elif pos < 90:
        sl.discard(value)
    elif pos < 99:
        _ = value in sl
    else:
        _iterate(sl)


@_register
def _multiset(sl: List, value: int, pos: int, add_fn: AddFn) -> None:
    if pos < 75:
        assert value in sl
    elif pos < 85:
        add_fn(sl, value)
    elif pos < 95:
        sl.remove(value)
    elif value < len(sl):
        _ = sl[value]


@_register
def _ranking(sl: List, value: int, pos: int, add_fn: AddFn) -> None:
    if pos < 40:
        if value < len(sl):
            _ = sl[value]
    elif pos < 80:
        assert sl.index(value) >= 0
    elif pos < 90:
        add_fn(sl, value)
    else:
        sl.remove(value)


@_register
def _neighbor(sl: List, value: int, pos: int, add_fn: AddFn) -> None:
    if pos < 75:
        _ = sl.bisect_right(value)
    elif pos < 85:
        add_fn(sl, value)
    elif pos < 95:
        sl.remove(value)
    else:
        _iterate(sl)


@_register
def _intervals(sl: List, value: int, pos: int, add_fn: AddFn) -> None:
    if pos < 30:
        _ = sl.bisect_right(value)
    elif pos < 50:
        if value < len(sl):
            _ = sl[value]
    elif pos < 70:
        if value < len(sl):
            del sl[value]
    elif pos < 80:
        list_length = len(sl)
        if value < list_length:
            stop = min(value + 100, list_length)
            other = RANDOM.randrange(value, stop)
            _ = sl[value:other]
    elif pos < 90:
        add_fn(sl, value)
    else:
        sl.discard(value)


def _iterate(sl: Iterable[object]) -> None:
    Iter(sl).take(102).pipe(consume)
