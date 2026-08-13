from collections.abc import Iterable

from pyochain.abc import PyoIterable

from . import checks
from ._utils import ImplIter, assert_iter_eq


# pyrefly: ignore [implicit-abstract-class]
class _PyFail(Iterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplIter, Iterable[int]): ...


# pyrefly: ignore [implicit-abstract-class]
class _PyoFail(PyoIterable[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplIter, PyoIterable[int]): ...


def test_iterable() -> None:

    checks.init_fail(_PyFail)
    # pyrefly: ignore [bad-instantiation]
    checks.iter_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert_iter_eq(iter(_PyOk()), iter(_PyoOk()))
