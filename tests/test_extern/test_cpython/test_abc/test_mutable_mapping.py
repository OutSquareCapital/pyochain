from __future__ import annotations

from collections.abc import Callable

import pytest

from pyochain import Dict, SetMut, Vec
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


Sorted = SortedDict[str, int]


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
    mymap = Sorted()
    mymap["red"] = 5
    assert isinstance(mymap.keys(), PyoSet)
    assert isinstance(mymap.keys(), PyoKeysView)
    assert isinstance(mymap.values(), PyoCollection)
    assert isinstance(mymap.values(), PyoValuesView)
    assert isinstance(mymap.items(), PyoSet)
    assert isinstance(mymap.items(), PyoItemsView)


type UnionFn[T] = Callable[[set[T]], SetMut[T]]


@pytest.mark.parametrize(
    "union_fn", (Sorted().keys().union, Sorted().keys().__or__), ids=name
)
def test_mutable_mapping_subclass_union(union_fn: UnionFn[str]) -> None:
    # Test issue 9214
    mymap = Sorted()
    mymap["red"] = 5
    z = union_fn({"orange"})
    assert isinstance(z, SetMut)
    _ = Vec(z)
    mymap["blue"] = 7  # Shouldn't affect 'z'
    assert z.iter().sort() == ["orange", "red"]


@pytest.mark.parametrize(
    "union_fn", (Sorted().items().union, Sorted().items().__or__), ids=name
)
def test_mutable_mapping_subclass_union_items(
    union_fn: UnionFn[tuple[str, int]],
) -> None:
    # Test issue 9214
    mymap = Sorted()
    mymap["red"] = 5
    oranges = ("orange", 3)
    x = union_fn({oranges})
    assert isinstance(x, SetMut)
    _ = Vec(x)
    mymap["blue"] = 7  # Shouldn't affect 'x'
    assert x == {oranges, ("red", 5)}
