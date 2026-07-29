from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec
from pyochain.abc import (
    PyoCollection,
    PyoMapping,
    PyoMutableMapping,
    PyoMutableSet,
    PyoSequence,
    PyoSet,
)

from ._utils import validate_abstract_methods

if TYPE_CHECKING:
    from collections.abc import Iterator


def mul(x: int) -> int:
    return x * 2


ERRS_NON_ITER: tuple[object, ...] = (None, 42, math.pi, 1j, mul)


@pytest.mark.parametrize("x", ERRS_NON_ITER)
def test_non_collections(x: object) -> None:
    # Check some non-collections
    assert not isinstance(x, PyoCollection)
    assert not issubclass(type(x), PyoCollection)


ERRS_ITER: tuple[object, ...] = (iter(b""), iter(bytearray()), (x for x in ()))


@pytest.mark.parametrize("x", ERRS_ITER)
def test_non_collection_iterable(x: object) -> None:
    # Check some non-collection iterables
    assert not isinstance(x, PyoCollection)
    assert not issubclass(type(x), PyoCollection)


OKS: tuple[PyoCollection[object], ...] = (
    Set(()),
    SetMut(()),
    Dict[object, object](()),
    Seq(()),
    Vec(()),
    Dict[object, object](()).keys(),
    Dict[object, object](()).items(),
    Dict[object, object](()).values(),
)


@pytest.mark.parametrize("x", OKS)
def test_collections(x: PyoCollection[object]) -> None:
    # Check some collections
    assert isinstance(x, PyoCollection)
    assert issubclass(type(x), PyoCollection)


def test_collections_is_subclasses() -> None:
    # Check also PyoMapping, PyoMutableMapping, etc.
    assert issubclass(PyoSequence, PyoCollection)
    assert issubclass(PyoMapping, PyoCollection)
    assert issubclass(PyoMutableMapping, PyoCollection)
    assert issubclass(PyoSet, PyoCollection)
    assert issubclass(PyoMutableSet, PyoCollection)
    assert issubclass(PyoSequence, PyoCollection)


def test_direct_subclassing() -> None:
    # Check direct subclassing

    class Col(PyoCollection[object]):
        @override
        def __iter__(self) -> Iterator[object]:
            return iter([])

        @override
        def __len__(self) -> int:
            return 0

        @override
        def __contains__(self, item: object) -> bool:
            return False

    class DerCol(Col):
        pass

    assert list(iter(Col())) == []
    assert not issubclass(list, Col)
    assert not issubclass(set, Col)
    assert not issubclass(float, Col)
    assert list(iter(DerCol())) == []
    assert not issubclass(list, DerCol)
    assert not issubclass(set, DerCol)
    assert not issubclass(float, DerCol)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoCollection, "__len__", "__iter__", "__contains__")


def test_collection_non_iterable() -> None:
    # Check sized container non-iterable (which is not PyoCollection) etc.

    class ColNoIter:
        def __len__(self) -> int:
            return 0

        def __contains__(self, item: object) -> bool:
            return False

    class ColNoSize:
        def __iter__(self) -> Iterator[object]:
            return iter([])

        def __contains__(self, item: object) -> bool:
            return False

    class ColNoCont:
        def __iter__(self) -> Iterator[object]:
            return iter([])

        def __len__(self) -> int:
            return 0

    assert not issubclass(ColNoIter, PyoCollection)
    assert not isinstance(ColNoIter(), PyoCollection)
    assert not issubclass(ColNoSize, PyoCollection)
    assert not isinstance(ColNoSize(), PyoCollection)
    assert not issubclass(ColNoCont, PyoCollection)
    assert not isinstance(ColNoCont(), PyoCollection)


def test_none_blocking() -> None:
    # Check None blocking

    class SizeBlock:
        def __iter__(self) -> Iterator[object]:
            return iter([])

        def __contains__(self) -> bool:  # ruff:ignore[unexpected-special-method-signature]
            return False

        __len__ = None  # pyright: ignore[reportUnannotatedClassAttribute]

    class IterBlock:
        def __len__(self) -> int:
            return 0

        def __contains__(self) -> bool:  # ruff:ignore[unexpected-special-method-signature]
            return True

        __iter__ = None  # pyright: ignore[reportUnannotatedClassAttribute]

    assert not issubclass(SizeBlock, PyoCollection)
    assert not isinstance(SizeBlock(), PyoCollection)
    assert not issubclass(IterBlock, PyoCollection)
    assert not isinstance(IterBlock(), PyoCollection)
    # Check None blocking in subclass

    class ColImpl:
        def __iter__(self) -> Iterator[object]:
            return iter([])

        def __len__(self) -> int:
            return 0

        def __contains__(self, item: object) -> bool:
            return False

    class NonCol(ColImpl):
        __contains__ = None  # pyright: ignore[reportUnannotatedClassAttribute, reportAssignmentType]

    assert not issubclass(NonCol, PyoCollection)
    assert not isinstance(NonCol(), PyoCollection)
