from collections import deque
from collections.abc import Callable, Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import pytest

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
PYO_SEQ = Seq(PY_TUP)
PYO_VEC = Vec(PY_LIST)
PYO_DEQUE = Deque(PY_DEQUE)
PYO_RANGE = Range(1, 4)
PYO_SET = Set(PY_FROZENSET)
PYO_SETMUT = SetMut(PY_SET)
PYO_DICT = Dict(PY_DICT)


COLLECTION_METHODS: list[Fn[Collection[object]]] = [
    lambda x: tuple(iter(x)),
    lambda a: 2 in a,
    len,
]
SET_METHODS: list[Fn[AbstractSet[int]]] = [
    lambda a: a | {4, 5},
    lambda a: a & {1, 2, 4},
    lambda a: a - {1, 2},
    lambda a: a ^ {1, 2, 4},
    lambda a: a < {1, 2, 4},
    lambda a: a <= PY_SET,
    lambda a: a > {1, 2},
    lambda a: a >= PY_SET,
    lambda a: a == PY_SET,
    lambda a: a != PY_SET,
]
MAPPING_METHODS: list[Fn[Mapping[int, int]]] = [
    lambda a: a.get(2),
    lambda a: a.keys(),
    lambda a: a.values(),
    lambda a: a.items(),
]

SEQUENCE_METHODS: list[Fn[Sequence[object]]] = [
    lambda a: a.count(2),
    lambda a: a.index(2),
    lambda a: a[1],  # noqa: FURB118
    lambda a: a[-1],  # noqa: FURB118
    lambda x: list(reversed(x)),
]

TUP_METHODS: list[Fn[tuple[int, ...]]] = [
    lambda a: a + (4, 5),  # noqa: RUF005
    lambda a: a * 2,
    lambda a: a < (1, 2, 4),
    lambda a: a <= PY_TUP,
    lambda a: a > (1, 2, 2),
    lambda a: a >= PY_TUP,
    lambda a: a == PY_TUP,
    lambda a: a != PY_TUP,
]
LIST_METHODS: list[Fn[list[int]]] = [
    lambda a: a + [4, 5],  # noqa: RUF005
    lambda a: a * 2,
    lambda a: a < [1, 2, 4],
    lambda a: a <= PY_LIST,
    lambda a: a > [1, 2, 2],
    lambda a: a >= PY_LIST,
    lambda a: a == PY_LIST,
    lambda a: a != PY_LIST,
]
DEQUE_METHODS: list[Fn[deque[int]]] = [
    lambda a: a + deque([4, 5]),
    lambda a: a * 2,
    lambda a: a < deque([1, 2, 4]),
    lambda a: a <= PY_DEQUE,
    lambda a: a > deque([1, 2, 2]),
    lambda a: a >= PY_DEQUE,
    lambda a: a == PY_DEQUE,
    lambda a: a != PY_DEQUE,
]


@pytest.mark.parametrize("method", COLLECTION_METHODS)
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
    )


@pytest.mark.parametrize("method", SET_METHODS)
def test_pyoset_methods(method: Fn[AbstractSet[int]]) -> None:
    assert (
        method(PY_SET) == method(PY_FROZENSET) == method(PYO_SET) == method(PYO_SETMUT)
    )


@pytest.mark.parametrize("method", MAPPING_METHODS)
def test_pyomapping_methods(method: Fn[Mapping[int, int]]) -> None:
    assert method(PY_DICT) == method(PYO_DICT)


@pytest.mark.parametrize("method", SEQUENCE_METHODS)
def test_pyosequence_methods(method: Fn[Sequence[object]]) -> None:
    assert (
        method(PY_TUP)
        == method(PYO_SEQ)
        == method(PYO_VEC)
        == method(PYO_DEQUE)
        == method(PYO_RANGE)
    )


@pytest.mark.parametrize("method", TUP_METHODS)
def test_seq_methods(method: Fn[tuple[int, ...]]) -> None:
    assert method(PY_TUP) == method(PYO_SEQ)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("method", LIST_METHODS)
def test_vec_methods(method: Fn[list[int]]) -> None:
    assert method(PY_LIST) == method(PYO_VEC)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("method", DEQUE_METHODS)
def test_deque_methods(method: Fn[deque[int]]) -> None:
    assert method(PY_DEQUE) == method(PYO_DEQUE)  # pyright: ignore[reportArgumentType]
