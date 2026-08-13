from collections import abc

import pytest

# Subclasshook of python ABCs raise error as soon as the class (not the instance!) is instantiated.
# We unfortunately cannot reciprocate this in Pyo3 ATM. Most access will raise `TypeError` tho, which is the same error as the ABC.
# For some reason, add, discard and insert are raising `AttributeError` instead of `TypeError`.
# This is probably due to the fact that they are the only methods who are not called by builtins.
CATCH_TYPE_ERROR = pytest.raises(TypeError)
CATCH_ATTRIBUTE_ERROR = pytest.raises(AttributeError)


def init_fail(obj: abc.Callable[[], object]) -> None:
    with CATCH_TYPE_ERROR:
        _ = obj()


def iter_fail(obj: abc.Iterable[int]) -> None:
    with CATCH_TYPE_ERROR:
        _ = iter(obj)


def next_fail(obj: abc.Iterator[int]) -> None:
    with CATCH_TYPE_ERROR:
        _ = next(obj)


def len_fail(obj: abc.Sized) -> None:
    with CATCH_TYPE_ERROR:
        _ = len(obj)


def contains_fail(obj: abc.Container[int]) -> None:
    with CATCH_TYPE_ERROR:
        _ = 1 in obj


def reversed_fail(obj: abc.Reversible[int]) -> None:
    with CATCH_TYPE_ERROR:
        _ = reversed(obj)


def getitem_fail(obj: abc.Sequence[int] | abc.Mapping[int, int]) -> None:
    with CATCH_TYPE_ERROR:
        _ = obj[0]


def setitem_fail(
    obj: abc.MutableSequence[int] | abc.MutableMapping[int, int],
) -> None:
    with CATCH_TYPE_ERROR:
        obj[0] = 1


def delitem_fail(
    obj: abc.MutableSequence[int] | abc.MutableMapping[int, int],
) -> None:
    with CATCH_TYPE_ERROR:
        del obj[0]


def insert_fail(obj: abc.MutableSequence[int]) -> None:
    with CATCH_ATTRIBUTE_ERROR:
        obj.insert(0, 1)


def add_fail(obj: abc.MutableSet[int]) -> None:
    with CATCH_ATTRIBUTE_ERROR:
        obj.add(1)


def discard_fail(obj: abc.MutableSet[int]) -> None:
    with CATCH_ATTRIBUTE_ERROR:
        obj.discard(1)
