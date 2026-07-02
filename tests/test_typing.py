from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyochain import Iter, Ok, Option, Result, Seq, Set, Some, Vec

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Container,
        Iterable,
        Iterator,
        MutableSequence,
        Reversible,
        Sequence,
        Sized,
    )

    from pyochain import Peekable
    from pyochain.abc import (
        PyoCollection,
        PyoContainer,
        PyoIterable,
        PyoIterator,
        PyoMutableSequence,
        PyoReversible,
        PyoSequence,
        PyoSet,
        PyoSized,
    )


@dataclass
class Animal:
    pass


@dataclass
class Dog(Animal):
    pass


def check_covariance() -> None:
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


def _iterable(x: Iterable[Animal]) -> Iterable[Animal]:
    return x


def _iterator(x: Iterator[Animal]) -> Iterator[Animal]:
    return x


def _sized(x: Sized) -> Sized:
    return x


def _reversible(x: Reversible[Animal]) -> Reversible[Animal]:
    return x


def _container(x: Container[Animal]) -> Container[Animal]:
    return x


def _collection(x: Collection[Animal]) -> Collection[Animal]:
    return x


def _sequence(x: Sequence[Animal]) -> Sequence[Animal]:
    return x


def _mutable_sequence(x: MutableSequence[Dog]) -> MutableSequence[Dog]:
    return x


def check_iterable_args() -> None:
    base: PyoIterable[Dog] = Iter(())
    canary: Iterable[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)
    _ = _iterator(canary)
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_iterator_args() -> None:
    base: PyoIterator[Dog] = Iter(())
    canary: Iterator[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)
    _ = _iterator(canary)
    _ = _sized(base)  # pyright: ignore[reportArgumentType]
    _ = _sized(canary)  # pyright: ignore[reportArgumentType]
    _ = _container(base)  # pyright: ignore[reportArgumentType]
    _ = _container(canary)  # pyright: ignore[reportArgumentType]
    _ = _reversible(base)  # pyright: ignore[reportArgumentType]
    _ = _reversible(canary)  # pyright: ignore[reportArgumentType]
    _ = _collection(base)  # pyright: ignore[reportArgumentType]
    _ = _collection(canary)  # pyright: ignore[reportArgumentType]
    _ = _sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _sequence(canary)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sized_args() -> None:
    base: PyoSized = Seq(())
    canary: Sized = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_reversible_args() -> None:
    base: PyoReversible[Dog] = Seq(())
    canary: Reversible[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_container_args() -> None:
    base: PyoContainer[Dog] = Seq(())
    canary: Container[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_collection_args() -> None:
    base: PyoCollection[Dog] = Seq(())
    canary: Collection[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_sequence_args() -> None:
    base: PyoSequence[Dog] = Seq(())
    canary: Sequence[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)  # pyright: ignore[reportArgumentType]
    _ = _mutable_sequence(canary)  # pyright: ignore[reportArgumentType]


def check_mutable_sequence_args() -> None:
    base: PyoMutableSequence[Dog] = Vec(())
    canary: MutableSequence[Dog] = base
    _ = _iterable(base)
    _ = _iterable(canary)
    _ = _iterator(base)  # pyright: ignore[reportArgumentType]
    _ = _iterator(canary)  # pyright: ignore[reportArgumentType]
    _ = _sized(base)
    _ = _sized(canary)
    _ = _container(base)
    _ = _container(canary)
    _ = _reversible(base)
    _ = _reversible(canary)
    _ = _collection(base)
    _ = _collection(canary)
    _ = _sequence(base)
    _ = _sequence(canary)
    _ = _mutable_sequence(base)
    _ = _mutable_sequence(canary)
