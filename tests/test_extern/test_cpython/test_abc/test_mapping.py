from __future__ import annotations

from typing import TYPE_CHECKING, override

import pytest

from pyochain import Dict, Seq
from pyochain.abc import PyoMapping
from pyochain.collections import SortedDict

from ._utils import validate_abstract_methods, validate_comparison

if TYPE_CHECKING:
    from collections.abc import Iterator

OKS = Seq((Dict, SortedDict))


@pytest.mark.parametrize("x", OKS, ids=OKS.iter().map(lambda x: x.__name__))
def test_mapping_ok(x: type[PyoMapping[object, object]]) -> None:
    # pyrefly: ignore [bad-argument-count]
    assert isinstance(x(()), PyoMapping)  # pyright: ignore[reportCallIssue]
    assert issubclass(x, PyoMapping)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(PyoMapping, "__iter__", "__len__", "__getitem__")


class MyMapping(PyoMapping[object, object]):
    @override
    def __len__(self) -> int:
        return 0

    @override
    def __getitem__(self, i: object) -> object:
        raise IndexError

    @override
    def __iter__(self) -> Iterator[object]:
        return iter(())


def test_subclass_comparison() -> None:
    validate_comparison(MyMapping())


def test_subclass_reversed_type_error() -> None:
    with pytest.raises(TypeError):
        _ = reversed(MyMapping())
