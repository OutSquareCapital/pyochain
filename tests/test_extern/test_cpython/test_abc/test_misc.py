from __future__ import annotations

import pytest

from pyochain.abc import (
    PyoCollection,
    PyoContainer,
    PyoIterable,
    PyoIterator,
    PyoMapping,
    PyoReversible,
    PyoSequence,
    PyoSized,
)

REGISTERABLES = (PyoIterable, PyoIterator, PyoReversible, PyoSized, PyoContainer)


@pytest.mark.parametrize("cls", REGISTERABLES)
def test_direct_subclassing(cls: type) -> None:

    class C(cls):  # pyright: ignore[reportUntypedBaseClass]
        pass

    assert issubclass(C, cls)
    assert not issubclass(int, C)


@pytest.mark.skip(
    reason="We don't support registering yet, as metaclasses aren't yet available with pyo3"
)
@pytest.mark.parametrize("cls", REGISTERABLES)
def test_registration(cls: type) -> None:
    class C:
        __hash__ = None  # Make sure it isn't hashable by default  # pyright: ignore[reportAssignmentType, reportUnannotatedClassAttribute]

    assert not issubclass(C, cls)
    cls.register(C)  # pyright: ignore[reportUnknownMemberType]
    assert issubclass(C, cls)


# From TestPyoCollectionABCs
# TODO: For now, we only test some virtual inheritance properties.
# We should also test the proper behavior of the collection ABCs
# as real base classes or mix-in classes.


def test_illegal_patma_flags() -> None:
    with pytest.raises(TypeError):

        class Both(PyoCollection[object]):  # pyright: ignore[reportImplicitAbstractClass, reportUnusedClass]
            __abc_tpflags__ = PyoSequence.__flags__ | PyoMapping.__flags__  # pyright: ignore[reportUnannotatedClassAttribute]
