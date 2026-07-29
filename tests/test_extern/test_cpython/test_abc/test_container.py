import math
from collections.abc import Container, Sequence
from typing import override

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec
from pyochain.abc import (
    PyoContainer,
    PyoItemsView,
    PyoKeysView,
    PyoSequence,
    PyoValuesView,
)

from ._utils import NEVER_EQ, validate_abstract_methods, validate_isinstance

ERRS: list[object] = [None, 42, math.pi, 1j, (x for x in ())]


@pytest.mark.parametrize("x", ERRS)
def test_not_pyocontainer(x: object) -> None:
    assert not isinstance(x, PyoContainer)
    assert not issubclass(type(x), PyoContainer)


OKS: Seq[object] = Seq((
    Seq(()),
    Vec(()),
    Set(()),
    SetMut(()),
    Dict[object, object](()),
    Dict[object, object](()).keys(),
    Dict[object, object](()).items(),
))


@pytest.mark.skip(reason="Same issue as with `Reversible` subclassing tests")
@pytest.mark.parametrize("x", OKS, ids=OKS.iter().map(lambda x: x.__class__.__name__))
def test_is_pyocontainer(x: PyoContainer[object]) -> None:
    assert isinstance(x, PyoContainer)
    assert issubclass(type(x), PyoContainer)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoContainer, "__contains__")


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_isinstance() -> None:
    validate_isinstance(PyoContainer, "__contains__")


# PyoContainer membership test should check identity first
class CustomSequence(PyoSequence[object]):
    def __init__(self, seq: Sequence[object]) -> None:
        self._seq: Sequence[object] = seq

    @override
    def __getitem__(self, index: int) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._seq[index]

    @override
    def __len__(self) -> int:
        return len(self._seq)


nan = float("nan")

CONTAINERS: list[Container[object]] = [
    CustomSequence([nan, NEVER_EQ, nan]),
    PyoItemsView({1: nan, 2: NEVER_EQ}),
    PyoKeysView({1: nan, 2: NEVER_EQ}),
    PyoValuesView({1: nan, 2: NEVER_EQ}),
]


def test_issue26915_seq() -> None:
    seq = CustomSequence([nan, NEVER_EQ, nan])
    assert seq.index(nan) == 0
    assert seq.index(NEVER_EQ) == 1
    assert seq.count(nan) == 2
    assert seq.count(NEVER_EQ) == 1


@pytest.mark.parametrize("container", CONTAINERS)
def test_issue26915(container: Container[object]) -> None:
    for elem in container:  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
        assert elem in container
