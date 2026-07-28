from collections.abc import Reversible

from pyochain.abc import PyoReversible

from . import checks
from ._utils import ImplRev, assert_iter_eq


class _PyFail(Reversible[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplRev, Reversible[int]): ...


class _PyoFail(PyoReversible[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplRev, PyoReversible[int]): ...


def test_reversible() -> None:
    checks.init_fail(_PyFail)
    checks.reversed_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert_iter_eq(reversed(_PyOk()), reversed(_PyoOk()))
