from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING, assert_type

from pyochain import Dict, Peekable, Seq, Set, Vec

from ._utils import Animal, Dog

if TYPE_CHECKING:
    from pyochain.abc import (
        PyoCollection,
        PyoContainer,
        PyoItemsView,
        PyoIterable,
        PyoIterator,
        PyoKeysView,
        PyoMapping,
        PyoMappingView,
        PyoMutableMapping,
        PyoMutableSequence,
        PyoReversible,
        PyoSequence,
        PyoSet,
        PyoSized,
        PyoValuesView,
    )


def check_dict() -> None:
    _ = assert_type(Dict({"a": 1, "b": 2}), Dict[str, int])
    # Avoid automatic literal inference
    data = [("a", 1), ("b", 2)]
    _ = assert_type(Dict(data), Dict[str, int])
    _ = assert_type(Dict(a=1, b=2), Dict[str, int])
    _ = assert_type(Dict({"a": 1}, b=2), Dict[str, int])


def check_iterables_covariance() -> None:
    base = Vec[Dog]
    _abc_iterable: PyoIterable[Animal] = base()
    _abc_iterator: PyoIterator[Animal] = base().iter()
    _abc_collection: PyoCollection[Animal] = base()
    _abc_sequence: PyoSequence[Animal] = base()
    _peekable_iterator: Peekable[Animal] = base().iter().peekable()
    _abc_set_immutable: PyoSet[Animal] = base().pipe(Set)
    _seq_immutable: Seq[Animal] = base().pipe(Seq)
    # pyrefly: ignore [bad-assignment]
    _: PyoMutableSequence[Animal] = base()  # pyright: ignore[reportAssignmentType]
    # pyrefly: ignore [bad-assignment]
    _: Vec[Animal] = base()  # pyright: ignore[reportAssignmentType]


type EntryData = list[tuple[object, tuple[str, ...]]]


def covariance_pyomapping(data: EntryData) -> None:
    d = Dict[object, abc.Sequence[object]](data)
    _ = assert_type(d, Dict[object, abc.Sequence[object]])


def check_iterable_args(base: PyoIterable[Dog], canary: abc.Iterable[Dog]) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sized(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _container(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_iterator_args(base: PyoIterator[Dog], canary: abc.Iterator[Dog]) -> None:
    _iterable(base, canary)
    _iterator(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sized(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _container(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_sized_args(base: PyoSized, canary: abc.Sized) -> None:
    # pyrefly: ignore [bad-argument-type]
    _iterable(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _container(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_reversible_args(
    base: PyoReversible[Dog], canary: abc.Reversible[Dog]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sized(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _container(base, canary)  # pyright: ignore[reportArgumentType]
    _reversible(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_container_args(
    base: PyoContainer[Animal], canary: abc.Container[Animal]
) -> None:
    # pyrefly: ignore [bad-argument-type]
    _iterable(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sized(base, canary)  # pyright: ignore[reportArgumentType]
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_collection_args(
    base: PyoCollection[Dog], canary: abc.Collection[Dog]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    _collection(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_sequence_args(base: PyoSequence[Dog], canary: abc.Sequence[Dog]) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    _reversible(base, canary)
    _collection(base, canary)
    _sequence(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]


def check_mutable_sequence_args(
    base: PyoMutableSequence[Dog], canary: abc.MutableSequence[Dog]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    _reversible(base, canary)
    _collection(base, canary)
    _sequence(base, canary)
    _mutable_sequence(base, canary)


def check_mapping(
    base: PyoMapping[Animal, Animal], canary: abc.Mapping[Animal, Animal]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    _collection(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _ = _mutable_mapping(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mapping_view(base, canary)  # pyright: ignore[reportArgumentType]


def check_mutable_mapping(
    base: PyoMutableMapping[Animal, Animal], canary: abc.MutableMapping[Animal, Animal]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    _collection(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping(base, canary)
    _ = _mutable_mapping(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mapping_view(base, canary)  # pyright: ignore[reportArgumentType]


def check_mapping_view_args(base: PyoMappingView, canary: abc.MappingView) -> None:
    # pyrefly: ignore [bad-argument-type]
    _iterable(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _container(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping_view(base, canary)


def check_items_view_args(
    base: PyoItemsView[Animal, Animal], canary: abc.ItemsView[Animal, Animal]
) -> None:
    # pyrefly: ignore [bad-argument-type]
    _iterable(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _collection(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping_view(base, canary)
    _items_view(base, canary)


def check_values_view_args(
    base: PyoValuesView[Animal], canary: abc.ValuesView[Animal]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    _collection(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping_view(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _keys_view(base, canary)  # pyright: ignore[reportArgumentType]
    _values_view(base, canary)


def check_keys_view_args(
    base: PyoKeysView[Animal], canary: abc.KeysView[Animal]
) -> None:
    _iterable(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _iterator(base, canary)  # pyright: ignore[reportArgumentType]
    _sized(base, canary)
    _container(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _reversible(base, canary)  # pyright: ignore[reportArgumentType]
    _collection(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _sequence(base, canary)  # pyright: ignore[reportArgumentType]
    # pyrefly: ignore [bad-argument-type]
    _mutable_sequence(base, canary)  # pyright: ignore[reportArgumentType]
    _mapping_view(base, canary)
    _keys_view(base, canary)
    # pyrefly: ignore [bad-argument-type]
    _values_view(base, canary)  # pyright: ignore[reportArgumentType]


def _iterable(*_: abc.Iterable[Animal]) -> None: ...
def _iterator(*_: abc.Iterator[Animal]) -> None: ...
def _sized(*_: abc.Sized) -> None: ...
def _container(*_: abc.Container[Animal]) -> None: ...
def _reversible(*_: abc.Reversible[Animal]) -> None: ...
def _collection(*_: abc.Collection[Animal]) -> None: ...
def _sequence(*_: abc.Sequence[Animal]) -> None: ...
def _mutable_sequence(*_: abc.MutableSequence[Dog]) -> None: ...
def _mapping_view(*_: abc.MappingView) -> None: ...
def _keys_view(*_: abc.KeysView[Animal]) -> None: ...
def _values_view(*_: abc.ValuesView[Animal]) -> None: ...
def _items_view(*_: abc.ItemsView[Animal, Animal]) -> None: ...
def _mapping(*_: abc.Mapping[Animal, Animal]) -> None: ...
def _mutable_mapping(*_: abc.MutableMapping[Animal, Animal]) -> None: ...
