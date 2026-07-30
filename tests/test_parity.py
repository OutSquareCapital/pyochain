from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import pytest
from _pytest.mark.structures import MarkDecorator

from pyochain import Dict, Range, Seq, Set, SetMut, Vec
from pyochain.collections import Deque

type Fn[T] = Callable[[T], Any]

PY_RANGE = range(1, 4)
PY_TUP = tuple[int, ...](PY_RANGE)
PY_LIST = list[int](PY_RANGE)
PY_DEQUE = deque[int](PY_RANGE)
PY_SET = set[int](PY_RANGE)
PY_FROZENSET = frozenset[int](PY_RANGE)
PY_DICT = dict[int, int]((i, i) for i in PY_RANGE)
PY_KEYS = PY_DICT.keys()
PY_VALUES = PY_DICT.values()
PY_ITEMS = PY_DICT.items()
PYO_SEQ = Seq(PY_TUP)
PYO_VEC = Vec(PY_LIST)
PYO_DEQUE = Deque(PY_DEQUE)
PYO_RANGE = Range(1, 4)
PYO_SET = Set(PY_FROZENSET)
PYO_SETMUT = SetMut(PY_SET)
PYO_DICT = Dict(PY_DICT)
PYO_KEYS = PYO_DICT.keys()
PYO_VALUES = PYO_DICT.values()
PYO_ITEMS = PYO_DICT.items()

type Params[T] = Dict[str, Fn[T]]


COLLECTION_METHODS: Params[Collection[object]] = Dict({
    "iter": lambda x: tuple(iter(x)),
    "contains": lambda a: 2 in a,
    "len": len,
})
SET_METHODS: Params[AbstractSet[int]] = Dict({
    "or": lambda a: a | {4, 5},
    "and": lambda a: a & {1, 2, 4},
    "sub": lambda a: a - {1, 2},
    "xor": lambda a: a ^ {1, 2, 4},
    "lt": lambda a: a < {1, 2, 4},
    "le": lambda a: a <= PY_SET,
    "gt": lambda a: a > {1, 2},
    "ge": lambda a: a >= PY_SET,
    "eq": lambda a: a == PY_SET,
    "ne": lambda a: a != PY_SET,
})
MAPPING_METHODS: Params[Mapping[int, int]] = Dict({
    "get": lambda a: a.get(2),
    "keys": lambda a: tuple(a.keys()),
    "values": lambda a: tuple(a.values()),
    "items": lambda a: tuple(a.items()),
})
KEYS_VIEW_METHODS: Params[Mapping[int, int]] = Dict({
    "contains": lambda a: 2 in a.keys(),  # ruff:ignore[in-dict-keys]
    "len": lambda a: len(a.keys()),
    "iter": lambda a: tuple(iter(a.keys())),
})
VALUES_VIEW_METHODS: Params[Mapping[int, int]] = Dict({
    "contains": lambda a: 2 in a.values(),
    "len": lambda a: len(a.values()),
    "iter": lambda a: tuple(iter(a.values())),
})
ITEMS_VIEW_METHODS: Params[Mapping[int, int]] = Dict({
    "contains": lambda a: (2, 2) in a.items(),
    "len": lambda a: len(a.items()),
    "iter": lambda a: tuple(iter(a.items())),
})

SEQUENCE_METHODS: Params[Sequence[object]] = Dict({
    "count": lambda a: a.count(2),
    "index": lambda a: a.index(2),
    "get_item_1": lambda a: a[1],  # ruff:ignore[reimplemented-operator]
    "get_item_neg_1": lambda a: a[-1],  # ruff:ignore[reimplemented-operator]
    "reversed": lambda x: tuple(reversed(x)),
})

TUP_METHODS: Params[tuple[int, ...]] = Dict({
    "add": lambda a: a + (4, 5),  # ruff:ignore[collection-literal-concatenation]
    "mul": lambda a: a * 2,
    "lt": lambda a: a < (1, 2, 4),
    "le": lambda a: a <= PY_TUP,
    "gt": lambda a: a > (1, 2, 2),
    "ge": lambda a: a >= PY_TUP,
    "eq": lambda a: a == PY_TUP,
    "ne": lambda a: a != PY_TUP,
})
LIST_METHODS: Params[list[int]] = Dict({
    "add": lambda a: a + [4, 5],  # ruff:ignore[collection-literal-concatenation]
    "mul": lambda a: a * 2,
    "lt": lambda a: a < [1, 2, 4],
    "le": lambda a: a <= PY_LIST,
    "gt": lambda a: a > [1, 2, 2],
    "ge": lambda a: a >= PY_LIST,
    "eq": lambda a: a == PY_LIST,
    "ne": lambda a: a != PY_LIST,
})
DEQUE_METHODS: Params[deque[int]] = Dict({
    "add": lambda a: a + deque([4, 5]),
    "mul": lambda a: a * 2,
    "lt": lambda a: a < deque([1, 2, 4]),
    "le": lambda a: a <= PY_DEQUE,
    "gt": lambda a: a > deque([1, 2, 2]),
    "ge": lambda a: a >= PY_DEQUE,
    "eq": lambda a: a == PY_DEQUE,
    "ne": lambda a: a != PY_DEQUE,
})


def _parametrize_methods[T](
    methods: Params[T],
) -> MarkDecorator:
    return pytest.mark.parametrize("method", methods.values(), ids=methods.keys())


@_parametrize_methods(COLLECTION_METHODS)
def test_pyocollection_methods(method: Fn[Collection[object]]) -> None:
    assert (
        method(PY_TUP)
        == method(PYO_SEQ)
        == method(PYO_VEC)
        == method(PYO_DEQUE)
        == method(PYO_RANGE)
        == method(PY_SET)
        == method(PY_FROZENSET)
        == method(PYO_SET)
        == method(PYO_SETMUT)
        == method(PYO_KEYS)
        == method(PY_KEYS)
        == method(PYO_VALUES)
        == method(PY_VALUES)
    )


@_parametrize_methods(SET_METHODS)
def test_pyoset_methods(method: Fn[AbstractSet[int]]) -> None:
    assert (
        method(PY_SET)
        == method(PY_FROZENSET)
        == method(PYO_SET)
        == method(PYO_SETMUT)
        == method(PYO_KEYS)
        == method(PY_KEYS)
    )


@_parametrize_methods(MAPPING_METHODS)
def test_pyomapping_methods(method: Fn[Mapping[int, int]]) -> None:
    assert method(PY_DICT) == method(PYO_DICT)


@_parametrize_methods(SEQUENCE_METHODS)
def test_pyosequence_methods(method: Fn[Sequence[object]]) -> None:
    assert (
        method(PY_TUP)
        == method(PYO_SEQ)
        == method(PYO_VEC)
        == method(PYO_DEQUE)
        == method(PYO_RANGE)
    )


@_parametrize_methods(TUP_METHODS)
def test_seq_methods(method: Fn[tuple[int, ...]]) -> None:
    # pyrefly: ignore [bad-argument-type]
    assert method(PY_TUP) == method(PYO_SEQ)  # pyright: ignore[reportArgumentType]


@_parametrize_methods(LIST_METHODS)
def test_vec_methods(method: Fn[list[int]]) -> None:
    # pyrefly: ignore [bad-argument-type]
    assert method(PY_LIST) == method(PYO_VEC)  # pyright: ignore[reportArgumentType]


@_parametrize_methods(DEQUE_METHODS)
def test_deque_methods(method: Fn[deque[int]]) -> None:
    # pyrefly: ignore [bad-argument-type]
    assert method(PY_DEQUE) == method(PYO_DEQUE)  # pyright: ignore[reportArgumentType]
