from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyochain import Iter, Ok, Option, Result, Seq, Set, Some

if TYPE_CHECKING:
    from pyochain._peekable import Peekable
    from pyochain.abc import (
        PyoCollection,
        PyoIterable,
        PyoIterator,
        PyoSequence,
        PyoSet,
    )


@dataclass
class Animal:
    pass


@dataclass
class Dog(Animal):
    pass


def check_covariance() -> None:
    """This can't be introspected at runtime. The type checker step is what will catch errors."""
    base: PyoIterable[Dog] = Iter(())
    opt: Option[Dog] = Some(Dog())
    res: Result[Dog, str] = Ok(Dog())
    _abc_iterable: PyoIterable[Animal] = base
    _abc_iterator: PyoIterator[Animal] = base
    _abc_collection: PyoCollection[Animal] = base.collect(Seq)
    _abc_sequence: PyoSequence[Animal] = base.collect(Seq)
    _concrete_iterator: Iter[Animal] = base
    _peekable_iterator: Peekable[Animal] = base.peekable()
    _abc_set_immutable: PyoSet[Animal] = base.collect(Set)
    _seq_immutable: Seq[Animal] = base.collect(Seq)
    _set_immutable: Set[Animal] = base.collect(Set)
    _as_opt: Option[Animal] = opt
    _as_res: Result[Animal, str] = res
