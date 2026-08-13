from collections.abc import Callable, Iterable, Iterator, MutableSet
from collections.abc import Set as AbstractSet
from typing import override

import pytest

from pyochain import Dict, Seq, Set, SetMut
from pyochain.abc import PyoMutableSet, PyoSet
from tests.test_abc._utils import ImplContainer, ImplSized

from . import checks
from ._utils import ImplCollection, ImplIter, assert_iter_eq

type SetPair[T] = tuple[AbstractSet[T], AbstractSet[T]]

type IntoSetMutFn = Callable[[set[int]], MutableSet[int]]


class _PyFail(AbstractSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplCollection, AbstractSet[int]): ...


class _PyoFail(PyoSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplCollection, PyoSet[int]): ...


def test_set() -> None:
    checks.init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    checks.len_fail(fail)
    checks.contains_fail(fail)
    checks.iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert len(py_ok) == len(pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    assert_iter_eq(py_ok, pyo_ok)
    assert py_ok <= pyo_ok
    assert py_ok >= pyo_ok
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # ruff:ignore[negate-not-equal-op]
    assert not (py_ok < pyo_ok)
    assert not (py_ok > pyo_ok)
    assert pyo_ok & py_ok == {1, 2, 3}
    assert pyo_ok | py_ok == {1, 2, 3}
    assert pyo_ok - py_ok == frozenset() == py_ok - pyo_ok
    assert pyo_ok ^ py_ok == frozenset() == py_ok ^ pyo_ok
    assert not pyo_ok.isdisjoint(py_ok)
    assert pyo_ok.isdisjoint(_PyOk([4, 5, 6]))


def test_set_from_iterable_note() -> None:
    class _NoIterableInit(ImplSized, ImplContainer, ImplIter): ...

    class _Bad(_NoIterableInit, PyoSet[int]): ...

    a = _Bad()
    b = _Bad()
    with pytest.raises(TypeError) as exc_info:
        _ = a | b
    assert exc_info.value.__notes__
    assert "PyoSet" in exc_info.value.__notes__[0]


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


def _get_all_sets() -> Dict[str, SetPair[int]]:
    data = (1, 2, 3)
    a = frozenset(data)
    b = ConcreteSet(data)
    c = set(data)
    d = Set(data)
    e = SetMut(data)
    f = ConcretePyoSet(data)
    return (
        Seq((a, b, c, d, e, f))
        .iter()
        .combinations(2)
        .map_star(
            lambda x, y: (f"{x.__class__.__name__} and {y.__class__.__name__}", (x, y))
        )
        .collect(Dict[str, SetPair[int]])
    )


ALL_SETS = _get_all_sets()


@pytest.mark.parametrize("sets", ALL_SETS.values(), ids=ALL_SETS.keys())
def test_and_dunder(sets: SetPair[int]) -> None:
    x, y = sets
    assert x == y
    assert x & y & y & x == x


@pytest.mark.parametrize("x_type", (SetMut, set, ConcretePyoSetMut))
@pytest.mark.parametrize("y_type", (SetMut, set, ConcretePyoSetMut))
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
    # set &= AbstractSet will return NotImplemented and then call AbstractSet.__rand__ which returns a new set, so the id will change.
    # This means that set &= SetMut won't be in-place (contrary to the opposite operation), UNLESS SetMut becomes itself a subclass of set.
    if (x_type is SetMut or x_type is ConcretePyoSetMut) and y_type is set:
        return

    assert id(y) == y_id
