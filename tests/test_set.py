from collections.abc import Iterable, Iterator
from collections.abc import Set as AbstractSet
from typing import override

from pyochain import Seq, Set, SetMut
from pyochain.abc import PyoSet


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
