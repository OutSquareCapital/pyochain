from collections.abc import Collection

from pyochain.abc import PyoCollection

from . import checks
from ._utils import ImplCollection, assert_iter_eq


# pyrefly: ignore [implicit-abstract-class]
class _PyFail(Collection[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyOk(ImplCollection, Collection[int]): ...


# pyrefly: ignore [implicit-abstract-class]
class _PyoFail(PyoCollection[int]): ...  # pyright: ignore[reportImplicitAbstractClass]


class _PyoOk(ImplCollection, PyoCollection[int]): ...


def test_collection() -> None:

    checks.init_fail(_PyFail)
    # pyrefly: ignore [bad-instantiation]
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
