"""Adapted from Cpython at https://github.com/python/cpython/blob/adf836ebddd89793afb6d43a79d0a0739ee5514b/Lib/test/test_collections.py"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import override

import pytest

from pyochain import Dict, SetMut
from pyochain.abc import (
    PyoCollection,
    PyoItemsView,
    PyoKeysView,
    PyoMutableMapping,
    PyoSet,
    PyoValuesView,
)
from pyochain.collections import SortedDict

from ._utils import validate_abstract_methods


def name(cls: type) -> str:
    return cls.__name__


class MyDict(PyoMutableMapping[str, int]):
    def __init__(self) -> None:
        self._data: dict[str, int] = {}

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    @override
    def __getitem__(self, key: str) -> int:
        return self._data[key]

    @override
    def __setitem__(self, key: str, value: int) -> None:
        self._data[key] = value

    @override
    def __delitem__(self, key: str) -> None:
        del self._data[key]

    @override
    def __len__(self) -> int:
        return len(self._data)


@pytest.mark.parametrize("cls", (Dict, SortedDict), ids=name)
def test_mutable_mapping(cls: type) -> None:
    assert isinstance(cls(()), PyoMutableMapping)
    assert issubclass(cls, PyoMutableMapping)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_abstract_methods() -> None:
    validate_abstract_methods(
        PyoMutableMapping,
        "__iter__",
        "__len__",
        "__getitem__",
        "__setitem__",
        "__delitem__",
    )


def test_mutable_mapping_subclass() -> None:
    # Test issue 9214
    mymap = MyDict()
    mymap["red"] = 5
    assert isinstance(mymap.keys(), PyoSet)
    assert isinstance(mymap.keys(), PyoKeysView)
    assert isinstance(mymap.values(), PyoCollection)
    assert isinstance(mymap.values(), PyoValuesView)
    assert isinstance(mymap.items(), PyoSet)
    assert isinstance(mymap.items(), PyoItemsView)


type UnionFn[T] = Callable[[set[T]], SetMut[T]]


def test_mutable_mapping_subclass_union() -> None:
    # Test issue 9214
    mymap = MyDict()
    mymap["red"] = 5
    z = mymap.keys() | {"orange"}
    assert isinstance(z, SetMut)
    _ = list(z)
    mymap["blue"] = 7  # Shouldn't affect 'z'
    assert z.iter().sort() == ["orange", "red"]


def test_mutable_mapping_subclass_union_items() -> None:
    # Test issue 9214

    mymap = MyDict()
    mymap["red"] = 5
    z = mymap.items() | {("orange", 3)}
    assert isinstance(z, SetMut)
    _ = list(z)
    mymap["blue"] = 7  # Shouldn't affect 'z'
    assert z == {("orange", 3), ("red", 5)}
