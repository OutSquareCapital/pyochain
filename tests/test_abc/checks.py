"""Subclasshook of python ABCs raise error as soon as the class (not the instance!) is instantiated.

We unfortunately cannot reciprocate this in Pyo3 ATM.

Most access will raise `TypeError` tho, which is the same error as the ABC.

## Note

Tests that can use builtin functions, reserved keywords, or operators, we call directly the method dunder, which will raise `AttributeError` instead of `TypeError`.
"""

from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections import abc


def init_fail(obj: abc.Callable[[], object]) -> None:
    _fail(partial(operator.call, obj), TypeError)


def iter_fail(obj: abc.Iterable[int]) -> None:
    _fail(partial(iter, obj), TypeError)


def next_fail(obj: abc.Iterator[int]) -> None:
    _fail(partial(next, obj), TypeError)


def len_fail(obj: abc.Sized) -> None:
    _fail(partial(len, obj), TypeError)


def contains_fail(obj: abc.Container[Any]) -> None:
    _fail(partial(operator.contains, obj, 0), TypeError)


def reversed_fail(obj: abc.Reversible[int]) -> None:
    _fail(partial(reversed, obj), TypeError)


def getitem_fail(obj: abc.Sequence[int] | abc.Mapping[int, int]) -> None:
    _fail(partial(operator.itemgetter(0), obj), TypeError)


def setitem_fail(
    obj: abc.MutableSequence[int] | abc.MutableMapping[int, int],
) -> None:
    _fail(lambda: obj.__setitem__(0, 1), AttributeError)


def delitem_fail(obj: abc.MutableSequence[int] | abc.MutableMapping[int, int]) -> None:
    _fail(lambda: obj.__delitem__(0), AttributeError)


def insert_fail(obj: abc.MutableSequence[int]) -> None:
    _fail(lambda: obj.insert(0, 1), AttributeError)


def add_fail(obj: abc.MutableSet[int]) -> None:
    _fail(lambda: obj.add(1), AttributeError)


def discard_fail(obj: abc.MutableSet[int]) -> None:
    _fail(lambda: obj.discard(1), AttributeError)


def _fail[T](method: abc.Callable[[], object], error: type[Exception]) -> None:
    with pytest.raises(error):
        _ = method()
