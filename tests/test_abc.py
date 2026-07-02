"""ABCs test module.

TODO: add comprehensive test suites for all ABCs, to:

    - Compare their behavior with the corresponding Python ABCs
    - Ensure errors on non-implemented methods
    - Confirm default implementations work as expected.
"""

from collections.abc import Container, Iterable, Iterator, Sized
from typing import override

import pytest

from pyochain.abc import PyoContainer, PyoIterable, PyoIterator, PyoSized

DATA = [1, 2, 3]


def _iter() -> Iterator[int]:
    return iter(DATA)


def test_iterable() -> None:
    class _PyFail(Iterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(Iterable[int]):
        @override
        def __iter__(self) -> Iterator[int]:
            return _iter()

    class _Pyo(PyoIterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(PyoIterable[int]):
        @override
        def __iter__(self) -> Iterator[int]:
            return _iter()

    # subclasshook of python Iterable raise error as soon as the class is instantiated
    with pytest.raises(TypeError):
        _ = _PyFail()  # pyright: ignore[reportAbstractUsage]
    # we can't do that (as far as I know) with Pyo3, so we check on the abstract method instead
    with pytest.raises(NotImplementedError):
        _ = iter(_Pyo())  # pyright: ignore[reportAbstractUsage]
    assert list(_PyOk()) == list(_PyoOk())


def test_iterator() -> None:
    class _PyFail(Iterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(Iterator[int]):
        def __init__(self) -> None:
            self._iter: Iterator[int] = _iter()

        @override
        def __next__(self) -> int:
            return next(self._iter)

    class _PyoFail(PyoIterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(PyoIterator[int]):
        def __init__(self) -> None:
            self._iter: Iterator[int] = _iter()

        @override
        def __next__(self) -> int:
            return next(self._iter)

    with pytest.raises(TypeError):
        _ = _PyFail()  # pyright: ignore[reportAbstractUsage]
    with pytest.raises(NotImplementedError):
        _ = next(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert list(_PyOk()) == list(_PyoOk())


def test_sized() -> None:
    class _PyFail(Sized): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(Sized):
        @override
        def __len__(self) -> int:
            return len(DATA)

    class _PyoFail(PyoSized): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(PyoSized):
        @override
        def __len__(self) -> int:
            return len(DATA)

    with pytest.raises(TypeError):
        _ = _PyFail()  # pyright: ignore[reportAbstractUsage]
    with pytest.raises(NotImplementedError):
        _ = len(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert len(_PyOk()) == len(_PyoOk())


def test_container() -> None:
    class _PyFail(Container[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyOk(Container[int]):
        @override
        def __contains__(self, item: int) -> bool:
            return item in DATA

    class _PyoFail(PyoContainer[int]): ...  # pyright: ignore[reportImplicitAbstractClass]

    class _PyoOk(PyoContainer[int]):
        @override
        def __contains__(self, item: int) -> bool:
            return item in DATA

    with pytest.raises(TypeError):
        _ = _PyFail()  # pyright: ignore[reportAbstractUsage]
    with pytest.raises(NotImplementedError):
        _ = 1 in _PyoFail()  # pyright: ignore[reportAbstractUsage]
    assert 1 in _PyOk()
    assert 1 in _PyoOk()
