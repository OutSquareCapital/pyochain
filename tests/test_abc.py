"""ABCs test module.

TODO: add comprehensive test suites for all ABCs, to:

    - Compare their behavior with the corresponding Python ABCs
    - Ensure errors on non-implemented methods
    - Confirm default implementations work as expected.
"""

from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Reversible,
    Sequence,
    Sized,
)
from collections.abc import (
    MutableSet as AbstractMutableSet,
)
from collections.abc import (
    Set as AbstractSet,
)

import pytest

from pyochain.abc import (
    PyoCollection,
    PyoContainer,
    PyoIterable,
    PyoIterator,
    PyoMapping,
    PyoMutableMapping,
    PyoMutableSequence,
    PyoMutableSet,
    PyoReversible,
    PyoSequence,
    PyoSet,
    PyoSized,
)

# Subclasshook of python ABCs raise error as soon as the class is instantiated.
CATCH_TYPE_ERROR = pytest.raises(TypeError)
# we can't do that (as far as I know) with Pyo3, so we check on the abstracts methods instead
CATCH_NOT_IMPLEMENTED = pytest.raises(NotImplementedError)


class _BaseImpl:
    def __init__(self) -> None:
        self._data: list[int] = [1, 2, 3]


class _ImplIter(_BaseImpl):
    def __iter__(self) -> Iterator[int]:
        return iter(self._data)


class _ImplSized(_BaseImpl):
    def __len__(self) -> int:
        return len(self._data)


class _ImplRev(_ImplIter):
    def __reversed__(self) -> Iterator[int]:
        return reversed(self._data)


class _ImplContainer(_BaseImpl):
    def __contains__(self, item: int) -> bool:
        return item in self._data


class _ImplCollection(_ImplSized, _ImplContainer, _ImplIter):
    """Works to implement both `Collection` and `Set` ABCs as a standlone."""


class _ImplMutableSet(_ImplCollection):
    def __init__(self) -> None:
        self._data: set[int] = {1, 2, 3}  # pyright: ignore[reportIncompatibleVariableOverride]

    def add(self, item: int) -> None:
        self._data.add(item)

    def discard(self, item: int) -> None:
        self._data.discard(item)


class _ImplSequence(_ImplSized):
    def __getitem__(self, index: int) -> int:
        return self._data[index]


class _ImplMapping(_ImplSequence, _ImplIter):
    def __init__(self) -> None:
        self._data: dict[int, int] = {i: i * 10 for i in [0, 1, 2]}  # pyright: ignore[reportIncompatibleVariableOverride]


class _ImplMutableMapping(_ImplMapping):
    def __setitem__(self, key: int, value: int) -> None:
        self._data[key] = value

    def __delitem__(self, key: int) -> None:
        del self._data[key]


class _ImplMutableSequence(_ImplSequence):
    def __setitem__(self, index: int, value: int) -> None:
        self._data[index] = value

    def __delitem__(self, index: int) -> None:
        del self._data[index]

    def insert(self, index: int, value: int) -> None:
        self._data.insert(index, value)


def test_iterable() -> None:

    class _PyFail(Iterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplIter, Iterable[int]): ...

    class _PyoFail(PyoIterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplIter, PyoIterable[int]): ...

    _check_abc_init_fail(_PyFail)
    _check_abc_iter_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    _assert_iter_eq(iter(_PyOk()), iter(_PyoOk()))


def test_iterator() -> None:
    class Impl:
        def __init__(self) -> None:
            self._iter: Iterator[int] = iter([1, 2, 3])

        def __next__(self) -> int:
            return next(self._iter)

    class _PyFail(Iterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(Impl, Iterator[int]): ...

    class _PyoFail(PyoIterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(Impl, PyoIterator[int]): ...

    _check_abc_init_fail(_PyFail)
    _check_abc_next_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert next(py_ok) == next(pyo_ok)
    _assert_iter_eq(py_ok, pyo_ok)


def test_sized() -> None:

    class _PyFail(Sized): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplSized, Sized): ...

    class _PyoFail(PyoSized): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplSized, PyoSized): ...

    _check_abc_init_fail(_PyFail)
    _check_abc_len_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert len(_PyOk()) == len(_PyoOk())


def test_container() -> None:

    class _PyFail(Container[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplContainer, Container[int]): ...

    class _PyoFail(PyoContainer[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplContainer, PyoContainer[int]): ...

    _check_abc_init_fail(_PyFail)
    _check_abc_contains_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert 1 in _PyOk()
    assert 1 in _PyoOk()


def test_reversible() -> None:

    class _PyFail(Reversible[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplRev, Reversible[int]): ...

    class _PyoFail(PyoReversible[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplRev, PyoReversible[int]): ...

    _check_abc_init_fail(_PyFail)
    _check_abc_reversed_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    _assert_iter_eq(reversed(_PyOk()), reversed(_PyoOk()))


def test_collection() -> None:
    class _PyFail(Collection[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplCollection, Collection[int]): ...

    class _PyoFail(PyoCollection[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplCollection, PyoCollection[int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_len_fail(fail)
    _check_abc_contains_fail(fail)
    _check_abc_iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert len(py_ok) == len(pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(py_ok, pyo_ok)


def test_set() -> None:
    class _PyFail(AbstractSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplCollection, AbstractSet[int]): ...

    class _PyoFail(PyoSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplCollection, PyoSet[int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_len_fail(fail)
    _check_abc_contains_fail(fail)
    _check_abc_iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert len(py_ok) == len(pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(py_ok, pyo_ok)
    assert py_ok <= pyo_ok
    assert py_ok >= pyo_ok
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # noqa: SIM202
    assert not (py_ok < pyo_ok)
    assert not (py_ok > pyo_ok)
    assert pyo_ok & py_ok == {1, 2, 3}
    assert pyo_ok | py_ok == {1, 2, 3}
    assert pyo_ok - py_ok == frozenset() == py_ok - pyo_ok
    assert pyo_ok ^ py_ok == frozenset() == py_ok ^ pyo_ok
    assert pyo_ok.isdisjoint(py_ok)


def test_mutable_set() -> None:  # noqa: PLR0915
    class _PyFail(AbstractMutableSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplMutableSet, AbstractMutableSet[int]): ...

    class _PyoFail(PyoMutableSet[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplMutableSet, PyoMutableSet[int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_len_fail(fail)
    _check_abc_contains_fail(fail)
    _check_abc_iter_fail(fail)
    _check_abc_add_fail(fail)
    _check_abc_discard_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert len(py_ok) == len(pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(py_ok, pyo_ok)
    assert py_ok <= pyo_ok
    assert py_ok >= pyo_ok
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # noqa: SIM202
    assert not (py_ok < pyo_ok)
    assert not (py_ok > pyo_ok)
    assert pyo_ok & py_ok == {1, 2, 3}
    assert pyo_ok | py_ok == {1, 2, 3}
    assert pyo_ok - py_ok == frozenset() == py_ok - pyo_ok
    assert pyo_ok ^ py_ok == frozenset() == py_ok ^ pyo_ok
    assert pyo_ok.isdisjoint(py_ok)
    pyo_ok.add(4)
    py_ok.add(4)
    assert 4 in pyo_ok
    assert 4 in py_ok
    pyo_ok.remove(4)
    py_ok.remove(4)
    assert 4 not in pyo_ok
    assert 4 not in py_ok
    pyo_ok.discard(5)
    py_ok.discard(5)
    additional = {4, 5}
    pyo_ok |= additional
    py_ok |= additional
    assert pyo_ok == py_ok
    pyo_ok &= {1, 2}
    py_ok &= {1, 2}
    assert pyo_ok == py_ok
    pyo_ok ^= {2, 3}
    py_ok ^= {2, 3}
    assert pyo_ok == py_ok
    pyo_ok -= {1}
    py_ok -= {1}
    assert pyo_ok == py_ok


def test_mapping() -> None:
    class _PyFail(Mapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplMapping, Mapping[int, int]): ...

    class _PyoFail(PyoMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplMapping, PyoMapping[int, int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_getitem_fail(fail)
    _check_abc_len_fail(fail)
    _check_abc_iter_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    _assert_iter_eq(py_ok, pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(py_ok.keys(), pyo_ok.keys())
    _assert_iter_eq(py_ok.values(), pyo_ok.values())
    _assert_iter_eq(py_ok.items(), pyo_ok.items())
    assert py_ok.get(0) == pyo_ok.get(0)
    assert py_ok == pyo_ok
    assert not (py_ok != pyo_ok)  # noqa: SIM202


def test_mutable_mapping() -> None:
    class _PyFail(MutableMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplMutableMapping, MutableMapping[int, int]): ...

    class _PyoFail(PyoMutableMapping[int, int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplMutableMapping, PyoMutableMapping[int, int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_getitem_fail(fail)
    _check_abc_len_fail(fail)
    _check_abc_iter_fail(fail)
    _check_abc_setitem_fail(fail)
    _check_abc_delitem_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    _assert_iter_eq(py_ok, pyo_ok)
    pyo_ok[0] = 100
    py_ok[0] = 100
    assert py_ok[0] == pyo_ok[0]
    del pyo_ok[0]
    del py_ok[0]
    assert py_ok == pyo_ok
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(py_ok.keys(), pyo_ok.keys())
    _assert_iter_eq(py_ok.values(), pyo_ok.values())
    _assert_iter_eq(py_ok.items(), pyo_ok.items())
    assert py_ok.get(0) == pyo_ok.get(0)
    assert not (py_ok != pyo_ok)  # noqa: SIM202
    assert py_ok.pop(1) == pyo_ok.pop(1)
    assert py_ok.popitem() == pyo_ok.popitem()
    assert py_ok.setdefault(0, 0) == pyo_ok.setdefault(0, 0)
    assert py_ok.update({0: 0}) == pyo_ok.update({0: 0})
    py_ok.clear()
    pyo_ok.clear()
    py_ok.update({0: 0})
    pyo_ok.update({0: 0})
    assert py_ok == pyo_ok


def test_sequence() -> None:
    class _PyFail(Sequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplSequence, Sequence[int]): ...

    class _PyoFail(PyoSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplSequence, PyoSequence[int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_getitem_fail(fail)
    _check_abc_len_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    _assert_iter_eq(py_ok, pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(reversed(py_ok), reversed(pyo_ok))
    assert py_ok.index(2) == pyo_ok.index(2)
    assert py_ok.count(2) == pyo_ok.count(2)


def test_mutable_sequence() -> None:
    class _PyFail(MutableSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(_ImplMutableSequence, MutableSequence[int]): ...

    class _PyoFail(PyoMutableSequence[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(_ImplMutableSequence, PyoMutableSequence[int]): ...

    _check_abc_init_fail(_PyFail)
    fail = _PyoFail()  # pyright: ignore[reportAbstractUsage]
    _check_abc_getitem_fail(fail)
    _check_abc_len_fail(fail)
    _check_abc_setitem_fail(fail)
    _check_abc_delitem_fail(fail)
    _check_abc_insert_fail(fail)
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert py_ok[0] == pyo_ok[0]
    assert len(py_ok) == len(pyo_ok)
    _assert_iter_eq(py_ok, pyo_ok)
    assert 1 in py_ok
    assert 1 in pyo_ok
    _assert_iter_eq(reversed(py_ok), reversed(pyo_ok))
    assert py_ok.index(2) == pyo_ok.index(2)
    assert py_ok.count(2) == pyo_ok.count(2)
    py_ok.append(4)
    pyo_ok.append(4)
    _assert_iter_eq(py_ok, pyo_ok)
    additional = [5, 6]
    pyo_ok.extend(additional)
    py_ok.extend(additional)
    assert pyo_ok.pop() == py_ok.pop()
    pyo_ok.remove(5)
    py_ok.remove(5)
    _assert_iter_eq(py_ok, pyo_ok)
    pyo_ok.reverse()
    py_ok.reverse()
    _assert_iter_eq(py_ok, pyo_ok)
    pyo_ok.clear()
    py_ok.clear()
    pyo_ok.insert(0, 1)
    py_ok.insert(0, 1)
    _assert_iter_eq(py_ok, pyo_ok)
    pyo_ok += additional
    py_ok += additional
    _assert_iter_eq(py_ok, pyo_ok)


def _assert_iter_eq(a: Iterable[object], b: Iterable[object]) -> None:
    assert tuple(a) == tuple(b)


def _check_abc_init_fail(obj: Callable[[], object]) -> None:
    with CATCH_TYPE_ERROR:
        _ = obj()


def _check_abc_iter_fail(obj: Iterable[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = iter(obj)


def _check_abc_next_fail(obj: Iterator[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = next(obj)


def _check_abc_len_fail(obj: Sized) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = len(obj)


def _check_abc_contains_fail(obj: Container[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = 1 in obj


def _check_abc_reversed_fail(obj: Reversible[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = reversed(obj)


def _check_abc_getitem_fail(obj: Sequence[int] | Mapping[int, int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        _ = obj[0]


def _check_abc_setitem_fail(
    obj: MutableSequence[int] | MutableMapping[int, int],
) -> None:
    with CATCH_NOT_IMPLEMENTED:
        obj[0] = 1


def _check_abc_delitem_fail(
    obj: MutableSequence[int] | MutableMapping[int, int],
) -> None:
    with CATCH_NOT_IMPLEMENTED:
        del obj[0]


def _check_abc_insert_fail(obj: MutableSequence[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        obj.insert(0, 1)


def _check_abc_add_fail(obj: AbstractMutableSet[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        obj.add(1)


def _check_abc_discard_fail(obj: AbstractMutableSet[int]) -> None:
    with CATCH_NOT_IMPLEMENTED:
        obj.discard(1)
