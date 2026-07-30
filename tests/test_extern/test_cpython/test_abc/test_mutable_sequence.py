from __future__ import annotations

import array
from typing import override

import pytest

from pyochain import Seq, Vec
from pyochain.abc import PyoMutableSequence
from pyochain.collections import Deque

from ._utils import validate_abstract_methods


def test_is_mutable_sequence() -> None:
    for sample in [Vec, Deque]:
        assert isinstance(sample(()), PyoMutableSequence)
        assert issubclass(sample, PyoMutableSequence)


def test_not_mutable_sequence() -> None:
    """NOTE: Contrary to python collections.abc.MutableSequence, PyoMutableSequence does not consider array.array or str to be a mutable sequence."""
    assert not issubclass(array.array, PyoMutableSequence)
    assert not issubclass(str, PyoMutableSequence)
    for sample in [Seq, str, bytes]:
        assert not isinstance(sample(()), PyoMutableSequence)
        assert not issubclass(sample, PyoMutableSequence)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(
        PyoMutableSequence,
        "__len__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
        "insert",
    )


def test_mutable_sequence_mixins() -> None:
    # Test the mixins of PyoMutableSequence by creating a minimal concrete
    # class inherited from it.
    class MutableSequenceSubclass(PyoMutableSequence[object]):
        def __init__(self) -> None:
            self.lst: list[object] = []

        @override
        # pyrefly: ignore [bad-override]
        def __setitem__(self, index: int, value: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
            self.lst[index] = value

        @override
        # pyrefly: ignore [bad-override]
        def __getitem__(self, index: int) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
            return self.lst[index]

        @override
        def __len__(self) -> int:
            return len(self.lst)

        @override
        # pyrefly: ignore [bad-override]
        def __delitem__(self, index: int) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
            del self.lst[index]

        @override
        def insert(self, index: int, value: object) -> None:
            self.lst.insert(index, value)

    mss = MutableSequenceSubclass()
    mss.append(0)
    mss.extend((1, 2, 3, 4))
    assert len(mss) == 5
    assert mss[3] == 3
    mss.reverse()
    assert mss[3] == 1
    _ = mss.pop()
    assert len(mss) == 4
    mss.remove(3)
    assert len(mss) == 3
    mss += (10, 20, 30)
    assert len(mss) == 6
    assert mss[-1] == 30
    mss.clear()
    assert len(mss) == 0

    # issue 34427
    # extending self should not cause infinite loop
    items = "ABCD"
    mss2 = MutableSequenceSubclass()
    mss2.extend(items + items)
    mss.clear()
    mss.extend(items)
    mss.extend(mss)
    assert len(mss) == len(mss2)
    assert list(mss) == list(mss2)
