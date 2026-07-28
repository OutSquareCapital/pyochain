from collections.abc import Container

from pyochain.abc import PyoContainer

from . import checks
from ._utils import ImplContainer


class _PyFail(Container[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplContainer, Container[int]): ...


class _PyoFail(PyoContainer[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplContainer, PyoContainer[int]): ...


def test_container() -> None:
    checks.init_fail(_PyFail)
    checks.contains_fail(_PyoFail())  # pyright: ignore[reportAbstractUsage]
    assert 1 in _PyOk()
    assert 1 in _PyoOk()
