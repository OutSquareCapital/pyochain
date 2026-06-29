from __future__ import annotations

from typing import TYPE_CHECKING

from pyochain import Iter

if TYPE_CHECKING:
    from pyochain._peekable import Peekable
    from pyochain.abc import PyoIterable, PyoIterator


class Animal: ...


class Dog(Animal): ...


def check_covariance() -> None:
    """This can't be introspected at runtime. The type checker step is what will catch errors."""
    base: PyoIterable[Dog] = Iter(())
    _abc_iterable: PyoIterable[Animal] = base
    _abc_iterator: PyoIterator[Animal] = base
    _concrete_iterator: Iter[Animal] = base
    _peekable_iterator: Peekable[Animal] = base.peekable()
