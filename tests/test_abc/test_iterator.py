from collections.abc import Iterator

from pyochain.abc import PyoIterator
from tests.test_abc._utils import assert_iter_eq

from . import checks


class Impl:
    def __init__(self) -> None:
        self._iter: Iterator[int] = iter([1, 2, 3])

    def __next__(self) -> int:
        return next(self._iter)


class _PyFail(Iterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(Impl, Iterator[int]): ...


class _PyoFail(PyoIterator[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(Impl, PyoIterator[int]): ...


def test_iterator() -> None:

    checks.init_fail(_PyFail)
    checks.next_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    py_ok = _PyOk()
    pyo_ok = _PyoOk()
    assert next(py_ok) == next(pyo_ok)
    assert_iter_eq(py_ok, pyo_ok)
