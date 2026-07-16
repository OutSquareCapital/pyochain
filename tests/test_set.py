from collections.abc import Callable, Iterable, Iterator, MutableSet
from collections.abc import Set as AbstractSet
from typing import override

import pytest

from pyochain import Seq, Set, SetMut
from pyochain.abc import PyoMutableSet, PyoSet


class ConcreteSet(AbstractSet[int]):
    def __init__(self, data: Iterable[int]) -> None:
        self.data: set[int] = set(data)

    @override
    def __repr__(self) -> str:
        return f"ConcreteSet({self.data})"

    @override
    def __contains__(self, item: object) -> bool:
        return item in self.data

    @override
    def __iter__(self) -> Iterator[int]:
        return iter(self.data)

    @override
    def __len__(self) -> int:
        return len(self.data)


class ConcretePyoSet(ConcreteSet, PyoSet[int]):
    pass


class ConcretePyoSetMut(ConcreteSet, PyoMutableSet[int]):
    @override
    def add(self, item: int) -> None:
        self.data.add(item)

    @override
    def discard(self, item: int) -> None:
        self.data.discard(item)


def test_and_dunder() -> None:
    data = (1, 2, 3)

    a = frozenset(data)
    b = ConcreteSet(data)
    c = set(data)
    d = Set(data)
    e = SetMut(data)
    f = ConcretePyoSet(data)
    assert a == c == d == e
    assert a == b == f
    assert d == e

    assert (
        Seq((a, b, c, d, e, f))
        .iter()
        .combinations(2)
        .map_star(lambda x, y: x & y & y & x == x)
        .all()
    )


type IntoSetMutFn = Callable[[set[int]], MutableSet[int]]


@pytest.mark.parametrize("x_type", [SetMut, set, ConcretePyoSetMut])
@pytest.mark.parametrize("y_type", [SetMut, set, ConcretePyoSetMut])
def test_iand_with_set(x_type: IntoSetMutFn, y_type: IntoSetMutFn) -> None:
    base = {1, 2, 3}
    x = x_type(base)
    y = y_type({2, 3, 4})
    x_id = id(x)
    y_id = id(y)
    y &= x
    assert x == base
    assert y == {2, 3}
    assert id(x) == x_id
    assert id(y) == y_id
