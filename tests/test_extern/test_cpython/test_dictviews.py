"""Adapted from CPython's test_dictviews.py to test the dict views in pyochain.

Original at:

https://github.com/python/cpython/blob/main/Lib/test/test_dictviews.py
"""

import copy
import pickle
from collections import abc
from typing import Literal, Never, override

import pytest

from pyochain import Dict, Seq, Set, SetMut, Vec


@pytest.mark.skip(
    reason="""We only have ABCs-level views for now, so this test is not applicable.
    In any case, why would this be even needed? Why restrict the constructor?"""
)
def test_constructors_not_callable() -> None:
    kt = type(Dict[object, object](()).keys())
    with pytest.raises(TypeError):
        _ = kt(Dict[object, object](()))
    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        _ = kt()  # pyright: ignore[reportCallIssue]
    it = type(Dict[object, object](()).items())
    with pytest.raises(TypeError):
        _ = it(Dict[object, object](()))
    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        _ = it()  # pyright: ignore[reportCallIssue]
    vt = type(Dict[object, object](()).values())
    with pytest.raises(TypeError):
        _ = vt(Dict[object, object](()))
    with pytest.raises(TypeError):
        # pyrefly: ignore [missing-argument]
        _ = vt()  # pyright: ignore[ reportCallIssue]


def test_dict_keys() -> None:
    d = Dict({1: 10, "a": "ABC"})
    keys = d.keys()
    assert len(keys) == 2
    assert SetMut(keys) == {1, "a"}
    assert keys == {1, "a"}
    assert keys != {1, "a", "b"}
    assert keys != {1, "b"}
    assert keys != {1}
    assert keys != 42
    assert 1 in keys
    assert "a" in keys
    assert 10 not in keys
    assert "Z" not in keys
    assert d.keys() == d.keys()
    e = Dict({1: 11, "a": "def"})
    assert d.keys() == e.keys()
    del e["a"]
    assert d.keys() != e.keys()


def test_dict_items() -> None:
    d = Dict({1: 10, "a": "ABC"})
    items = d.items()
    assert len(items) == 2
    assert SetMut(items) == {(1, 10), ("a", "ABC")}
    assert items == {(1, 10), ("a", "ABC")}
    assert items != {(1, 10), ("a", "ABC"), "junk"}
    assert items != {(1, 10), ("a", "def")}
    assert items != {(1, 10)}
    assert items != 42
    assert (1, 10) in items
    assert ("a", "ABC") in items
    assert (1, 11) not in items
    # pyrefly: ignore [unsupported-operation]
    assert 1 not in items  # pyright: ignore[reportOperatorIssue]
    # pyrefly: ignore [unsupported-operation]
    assert () not in items  # pyright: ignore[reportOperatorIssue]
    # pyrefly: ignore [unsupported-operation]
    assert (1,) not in items  # pyright: ignore[reportOperatorIssue]
    # pyrefly: ignore [unsupported-operation]
    assert (1, 2, 3) not in items  # pyright: ignore[reportOperatorIssue]
    assert d.items() == d.items()
    e = d.copy()
    assert d.items() == e.items()
    e["a"] = "def"
    assert d.items() != e.items()


def test_dict_mixed_keys_items() -> None:
    d = Dict({(1, 1): 11, (2, 2): 22})
    e = Dict({1: 1, 2: 2})
    assert d.keys() == e.items()
    assert d.items() != e.keys()


def test_dict_values() -> None:
    d = Dict({1: 10, "a": "ABC"})
    values = d.values()
    assert SetMut(values) == {10, "ABC"}
    assert len(values) == 2


def test_item_views_repr() -> None:
    d = Dict({1: 10, "a": "ABC"})
    assert isinstance(repr(d), str)
    r = repr(d.items())
    assert isinstance(r, str)
    assert r in {
        "PyoItemsView(Dict(1: 10, 'a': 'ABC'))",
        "PyoItemsView(Dict('a': 'ABC', 1: 10))",
    }


# NOTE: Our repr diverge from CPython's dict view, and is more closer to corresponding ABC's reprs.


def test_keys_view_repr() -> None:
    d = Dict({1: 10, "a": "ABC"})
    r = repr(d.keys())
    assert isinstance(r, str)
    assert r in {
        "PyoKeysView(Dict('a': 'ABC', 1: 10))",
        "PyoKeysView(Dict(1: 10, 'a': 'ABC'))",
    }


def test_values_view_repr() -> None:
    d = Dict({1: 10, "a": "ABC"})
    r = repr(d.values())
    assert isinstance(r, str)
    assert r in {
        "PyoValuesView(Dict(1: 10, 'a': 'ABC'))",
        "PyoValuesView(Dict('a': 'ABC', 1: 10))",
    }


def test_keys_and_set_operations() -> None:
    d1, d2, d3, d4 = _data_for_keys_set_operations()
    assert d1.keys() & d1.keys() == {"a", "b"}
    assert d1.keys() & d2.keys() == {"b"}
    assert d1.keys() & d3.keys() == SetMut(())
    assert d1.keys() & SetMut(d1.keys()) == {"a", "b"}
    assert d1.keys() & SetMut(d2.keys()) == {"b"}
    assert d1.keys() & SetMut(d3.keys()) == SetMut(())
    assert d1.keys() & tuple(d1.keys()) == {"a", "b"}
    assert d3.keys() & d4.keys() == {"d"}
    assert d4.keys() & d3.keys() == {"d"}
    assert d4.keys() & SetMut(d3.keys()) == {"d"}
    assert isinstance(d4.keys() & Set(d3.keys()), SetMut)
    set_and_keys = Set(d3.keys()) & d4.keys()
    # NOTE: Here we diverge from CPython -> frozenset & dict_keys will in fact call `dict_keys.__and__(frozenset)`, thus returning a `set`
    # This is NOT what's expected if one read the stubs from typeshed.
    # Thus, I took the decision to be consistent with typing and return a `Set` instead of a `SetMut`.
    # Basically, given `T.__and__(U)`, we always return T (unless T is not an AbstractSet, which is a TypeError).
    assert isinstance(set_and_keys, Set)
    assert not isinstance(set_and_keys, SetMut)
    assert type(d4.keys() & SetMut(d3.keys())) is SetMut
    assert type(d1.keys() & []) is SetMut
    assert type(Vec(()) & d1.keys()) is SetMut


def test_keys_and_set_operations_or() -> None:
    d1, d2, d3, _ = _data_for_keys_set_operations()
    assert d1.keys() | d1.keys() == {"a", "b"}
    assert d1.keys() | d2.keys() == {"a", "b", "c"}
    assert d1.keys() | d3.keys() == {"a", "b", "d", "e"}
    assert d1.keys() | SetMut(d1.keys()) == {"a", "b"}
    assert d1.keys() | SetMut(d2.keys()) == {"a", "b", "c"}
    assert d1.keys() | SetMut(d3.keys()) == {"a", "b", "d", "e"}
    assert d1.keys() | (1, 2) == {"a", "b", 1, 2}


def test_keys_and_set_operations_xor() -> None:
    d1, d2, d3, _ = _data_for_keys_set_operations()
    assert d1.keys() ^ d1.keys() == SetMut(())
    assert d1.keys() ^ d2.keys() == {"a", "c"}
    assert d1.keys() ^ d3.keys() == {"a", "b", "d", "e"}
    assert d1.keys() ^ SetMut(d1.keys()) == SetMut(())
    assert d1.keys() ^ SetMut(d2.keys()) == {"a", "c"}
    assert d1.keys() ^ SetMut(d3.keys()) == {"a", "b", "d", "e"}
    assert d1.keys() ^ tuple(d2.keys()) == {"a", "c"}


def test_keys_and_set_operations_sub() -> None:
    d1, d2, d3, _ = _data_for_keys_set_operations()
    assert d1.keys() - d1.keys() == SetMut(())
    assert d1.keys() - d2.keys() == {"a"}
    assert d1.keys() - d3.keys() == {"a", "b"}
    assert d1.keys() - SetMut(d1.keys()) == SetMut(())
    assert d1.keys() - SetMut(d2.keys()) == {"a"}
    assert d1.keys() - SetMut(d3.keys()) == {"a", "b"}
    assert d1.keys() - (0, 1) == {"a", "b"}


def test_keys_and_set_operations_isdisjoint() -> None:
    d1, d2, d3, _ = _data_for_keys_set_operations()
    assert not d1.keys().isdisjoint(d1.keys())
    assert not d1.keys().isdisjoint(d2.keys())
    assert not d1.keys().isdisjoint(list(d2.keys()))
    assert not d1.keys().isdisjoint(SetMut(d2.keys()))
    assert d1.keys().isdisjoint({"x", "y", "z"})
    assert d1.keys().isdisjoint(["x", "y", "z"])
    assert d1.keys().isdisjoint({"x", "y", "z"})
    assert d1.keys().isdisjoint({"x", "y"})
    assert d1.keys().isdisjoint(["x", "y"])
    assert d1.keys().isdisjoint({})
    assert d1.keys().isdisjoint(d3.keys())

    de = Dict[object, object](())
    assert de.keys().isdisjoint(SetMut(()))
    assert de.keys().isdisjoint([])
    assert de.keys().isdisjoint(de.keys())
    assert de.keys().isdisjoint([1])


def _data_for_keys_set_operations() -> tuple[
    Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]
]:
    d1 = Dict({"a": 1, "b": 2})
    d2 = Dict({"b": 3, "c": 2})
    d3 = Dict({"d": 4, "e": 5})
    d4 = Dict({"d": 4})
    return d1, d2, d3, d4


def test_items_and_set_and() -> None:
    d1, d2, d3 = _data_for_items_set_operations()
    assert d1.items() & d1.items() == {("a", 1), ("b", 2)}
    assert d1.items() & d2.items() == {("b", 2)}
    assert d1.items() & d3.items() == SetMut(())
    assert d1.items() & SetMut(d1.items()) == {("a", 1), ("b", 2)}
    assert d1.items() & SetMut(d2.items()) == {("b", 2)}
    assert d1.items() & SetMut(d3.items()) == SetMut(())
    assert d1.items() & (("a", 1), ("b", 2)) == {("a", 1), ("b", 2)}
    assert d1.items() & (("a", 2), ("b", 2)) == {("b", 2)}
    assert d1.items() & (("d", 4), ("e", 5)) == SetMut(())


def test_items_and_set_operations_or() -> None:
    d1, d2, d3 = _data_for_items_set_operations()

    assert d1.items() | d1.items() == {("a", 1), ("b", 2)}
    assert d1.items() | d2.items() == {("a", 1), ("a", 2), ("b", 2)}
    assert d1.items() | d3.items() == {("a", 1), ("b", 2), ("d", 4), ("e", 5)}
    assert d1.items() | SetMut(d1.items()) == {("a", 1), ("b", 2)}
    assert d1.items() | SetMut(d2.items()) == {("a", 1), ("a", 2), ("b", 2)}
    assert d1.items() | SetMut(d3.items()) == {("a", 1), ("b", 2), ("d", 4), ("e", 5)}
    assert d1.items() | (("a", 1), ("b", 2)) == {("a", 1), ("b", 2)}
    assert d1.items() | (("a", 2), ("b", 2)) == {("a", 1), ("a", 2), ("b", 2)}
    assert d1.items() | (("d", 4), ("e", 5)) == {("a", 1), ("b", 2), ("d", 4), ("e", 5)}


def test_items_and_set_operations_xor() -> None:
    d1, d2, d3 = _data_for_items_set_operations()
    assert d1.items() ^ d1.items() == SetMut(())
    assert d1.items() ^ d2.items() == {("a", 1), ("a", 2)}
    assert d1.items() ^ d3.items() == {("a", 1), ("b", 2), ("d", 4), ("e", 5)}
    assert d1.items() ^ (("a", 1), ("b", 2)) == SetMut(())
    assert d1.items() ^ (("a", 2), ("b", 2)) == {("a", 1), ("a", 2)}
    assert d1.items() ^ (("d", 4), ("e", 5)) == {("a", 1), ("b", 2), ("d", 4), ("e", 5)}


def test_items_and_set_operations_sub() -> None:
    d1, d2, d3 = _data_for_items_set_operations()
    assert d1.items() - d1.items() == SetMut(())
    assert d1.items() - d2.items() == {("a", 1)}
    assert d1.items() - d3.items() == {("a", 1), ("b", 2)}
    assert d1.items() - SetMut(d1.items()) == SetMut(())
    assert d1.items() - SetMut(d2.items()) == {("a", 1)}
    assert d1.items() - SetMut(d3.items()) == {("a", 1), ("b", 2)}
    assert d1.items() - (("a", 1), ("b", 2)) == SetMut(())
    assert d1.items() - (("a", 2), ("b", 2)) == {("a", 1)}
    assert d1.items() - (("d", 4), ("e", 5)) == {("a", 1), ("b", 2)}


def test_items_and_set_operations_isdisjoint() -> None:
    d1, d2, d3 = _data_for_items_set_operations()
    assert not d1.items().isdisjoint(d1.items())
    assert not d1.items().isdisjoint(d2.items())
    assert not d1.items().isdisjoint(list(d2.items()))
    assert not d1.items().isdisjoint(SetMut(d2.items()))
    assert d1.items().isdisjoint({"x", "y", "z"})
    assert d1.items().isdisjoint(["x", "y", "z"])
    assert d1.items().isdisjoint({"x", "y", "z"})
    assert d1.items().isdisjoint({"x", "y"})
    assert d1.items().isdisjoint({})
    assert d1.items().isdisjoint(d3.items())

    de = Dict[object, object](())
    assert de.items().isdisjoint(SetMut(()))
    assert de.items().isdisjoint([])
    assert de.items().isdisjoint(de.items())
    assert de.items().isdisjoint([1])


def _data_for_items_set_operations() -> tuple[
    Dict[str, int], Dict[str, int], Dict[str, int]
]:
    d1 = Dict({"a": 1, "b": 2})
    d2 = Dict({"a": 2, "b": 2})
    d3 = Dict({"d": 4, "e": 5})
    return d1, d2, d3


def test_keys_set_operations_with_iterator() -> None:
    origin = Dict({1: 2, 3: 4})
    other = Seq((1, 2))
    assert origin.keys() & other.iter() == {1}
    assert origin.keys() | other.iter() == {1, 2, 3}
    assert origin.keys() ^ other.iter() == {2, 3}
    assert origin.keys() - other.iter() == {3}


def test_items_set_operations_with_iterator() -> None:
    origin = Dict({1: 2, 3: 4})
    items = origin.items()
    other = Seq([(1, 2)])
    assert items & other.iter() == {(1, 2)}
    assert items ^ other.iter() == {(3, 4)}
    assert items | other.iter() == {(1, 2), (3, 4)}
    assert items - other.iter() == {(3, 4)}


def test_set_operations_with_noniterable() -> None:
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).keys() & 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).keys() | 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).keys() ^ 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).keys() - 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]

    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).items() & 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).items() | 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).items() ^ 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]
    with pytest.raises(TypeError):
        # pyrefly: ignore [unsupported-operation]
        _ = Dict[object, object](()).items() - 1  # pyright: ignore[reportOperatorIssue, reportUnknownVariableType]


@pytest.mark.skip(
    reason="We don't seem to handle recursive repr correctly as of now. Need to investigate further."
)
def test_recursive_repr() -> None:
    d = Dict[object, object](())
    d[42] = d.values()
    r = repr(d)
    # Cannot perform a stronger test, as the contents of the repr
    # are implementation-dependent.  All we can say is that we
    # want a str result, not an exception of any sort.
    assert isinstance(r, str)
    d[42] = d.items()
    r = repr(d)
    # Again.
    assert isinstance(r, str)


def test_copy() -> None:
    d = Dict({1: 10, "a": "ABC"})
    with pytest.raises(TypeError):
        _ = copy.copy(d.keys())
    with pytest.raises(TypeError):
        _ = copy.copy(d.values())
    with pytest.raises(TypeError):
        _ = copy.copy(d.items())


def test_compare_error() -> None:
    class ExcError(Exception):
        pass

    class BadEq:
        @override
        def __hash__(self) -> Literal[7]:
            return 7

        @override
        def __eq__(self, other: object) -> Never:
            raise ExcError

    k1, k2 = BadEq(), BadEq()
    v1, v2 = BadEq(), BadEq()
    d = Dict({k1: v1})

    assert k1 in d
    assert k1 in d
    assert v1 in d.values()
    assert (k1, v1) in d.items()

    with pytest.raises(ExcError):
        _ = d.__contains__(k2)
    with pytest.raises(ExcError):
        _ = d.keys().__contains__(k2)
    with pytest.raises(ExcError):
        _ = d.items().__contains__((k2, v1))
    with pytest.raises(ExcError):
        _ = d.items().__contains__((k1, v2))
    with pytest.raises(ExcError):
        _ = v2 in d.values()


def test_pickle() -> None:
    d = Dict({1: 10, "a": "ABC"})
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises((TypeError, pickle.PicklingError)):
            _ = pickle.dumps(d.keys(), proto)
        with pytest.raises((TypeError, pickle.PicklingError)):
            _ = pickle.dumps(d.values(), proto)
        with pytest.raises((TypeError, pickle.PicklingError)):
            _ = pickle.dumps(d.items(), proto)


def test_abc_registry_keys() -> None:
    d = Dict({"a": 1})

    assert isinstance(d.keys(), abc.KeysView)
    assert isinstance(d.keys(), abc.MappingView)
    assert isinstance(d.keys(), abc.Set)
    assert isinstance(d.keys(), abc.Sized)
    assert isinstance(d.keys(), abc.Iterable)
    assert isinstance(d.keys(), abc.Container)


def test_abc_registry_values() -> None:
    d = Dict({"a": 1})
    assert isinstance(d.values(), abc.ValuesView)
    assert isinstance(d.values(), abc.MappingView)
    assert isinstance(d.values(), abc.Sized)
    assert isinstance(d.values(), abc.Collection)
    assert isinstance(d.values(), abc.Iterable)
    assert isinstance(d.values(), abc.Container)


def test_abc_registry_items() -> None:
    d = Dict({"a": 1})
    assert isinstance(d.items(), abc.ItemsView)
    assert isinstance(d.items(), abc.MappingView)
    assert isinstance(d.items(), abc.Set)
    assert isinstance(d.items(), abc.Sized)
    assert isinstance(d.items(), abc.Iterable)
    assert isinstance(d.items(), abc.Container)
