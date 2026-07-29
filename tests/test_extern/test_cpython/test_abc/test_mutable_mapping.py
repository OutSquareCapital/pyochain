from __future__ import annotations

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


def test_mutable_mapping() -> None:
    for sample in [Dict, SortedDict]:
        assert isinstance(sample(()), PyoMutableMapping)
        assert issubclass(sample, PyoMutableMapping)


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
    mymap = SortedDict[str, int]()
    mymap["red"] = 5
    assert isinstance(mymap.keys(), PyoSet)
    assert isinstance(mymap.keys(), PyoKeysView)
    assert isinstance(mymap.values(), PyoCollection)
    assert isinstance(mymap.values(), PyoValuesView)
    assert isinstance(mymap.items(), PyoSet)
    assert isinstance(mymap.items(), PyoItemsView)


def test_mutable_mapping_subclass_union() -> None:
    # Test issue 9214
    mymap = SortedDict[str, int]()
    mymap["red"] = 5
    z = mymap.keys().union({"orange"})
    assert isinstance(z, SetMut)
    _ = list(z)
    mymap["blue"] = 7  # Shouldn't affect 'z'
    assert sorted(z) == ["orange", "red"]


def test_mutable_mapping_subclass_union_items() -> None:
    # Test issue 9214
    mymap = SortedDict[str, int]()
    mymap["red"] = 5
    z = mymap.items().union({("orange", 3)})
    assert isinstance(z, SetMut)
    _ = list(z)
    mymap["blue"] = 7  # Shouldn't affect 'z'
    assert z == {("orange", 3), ("red", 5)}
