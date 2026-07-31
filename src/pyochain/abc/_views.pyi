from collections.abc import (
    ItemsView,
    Iterable,
    Iterator,
    KeysView,
    MappingView,
    Sized,
    ValuesView,
)
from typing import Any, Generic, Self, TypeVar, override

from _typeshed import SupportsGetItemViewable, Viewable

from pyochain import SetMut
from pyochain.abc import PyoCollection, PyoSet, PyoSized

# NOTE: We are forced to use legacy `TypeVar` syntax here due to concrete limitations of the typing system. Typeshed explicitely ignore some typing warnings.
# https://github.com/python/typing/pull/273
_K_co = TypeVar("_K_co", covariant=True)
_V_co = TypeVar("_V_co", covariant=True)

class PyoMappingView(MappingView, PyoSized):
    """Extends both `MappingView` from `collections.abc` and `PyoCollection[T]`.

    Is the base class shared by the views returned by `PyoMapping` methods.
    """

    _mapping: Sized
    def __init__(self, mapping: Sized) -> None: ...
    @override
    def __len__(self) -> int: ...

class PyoKeysView(PyoMappingView, PyoSet[_K_co], KeysView[_K_co]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the keys in a pyochain mapping.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    def __init__(self, mapping: Viewable[_K_co]) -> None: ...
    @classmethod
    @override
    def _from_iterable[S](cls, it: Iterable[S], /) -> SetMut[S]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __contains__(self, key: object, /) -> bool: ...
    @override
    def __iter__(self) -> Iterator[_K_co]: ...
    @override
    def __and__(self, other: Iterable[Any], /) -> SetMut[_K_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rand__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __or__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ror__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __sub__[T](self, other: Iterable[Any], /) -> SetMut[_K_co]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rsub__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __xor__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rxor__[T](self, other: Iterable[T], /) -> SetMut[_K_co | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[_K_co]: ...
    @override
    def union[S, T](self: PyoKeysView[S], other: Iterable[T]) -> SetMut[S | T]: ...
    @override
    def difference(self, other: Iterable[Any]) -> SetMut[_K_co]: ...
    @override
    def symmetric_difference[S, T](
        self: PyoKeysView[S], other: Iterable[T]
    ) -> SetMut[S | T]: ...

class PyoValuesView[V](PyoMappingView, PyoCollection[V], ValuesView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """
    def __init__(self, mapping: SupportsGetItemViewable[Any, V]) -> None: ...
    @override
    def __contains__(self, value: object, /) -> bool: ...
    @override
    def __iter__(self) -> Iterator[V]: ...

class PyoItemsView(  # pyright: ignore[reportUnsafeMultipleInheritance]
    PyoMappingView,
    PyoSet[tuple[_K_co, _V_co]],
    ItemsView[_K_co, _V_co],
    Generic[_K_co, _V_co],  # ruff:ignore[non-pep695-generic-class]
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping::items`: Method that returns this view.
    """
    def __new__(cls, mapping: SupportsGetItemViewable[_K_co, _V_co]) -> Self: ...  # pyright: ignore[reportInconsistentConstructor]
    @classmethod
    @override
    def _from_iterable[S](cls, it: Iterable[S], /) -> set[S]: ...
    @override
    # pyrefly: ignore [bad-override]
    def __contains__(self, item: tuple[object, object], /) -> bool: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __iter__(self) -> Iterator[tuple[_K_co, _V_co]]: ...
    @override
    def __and__(self, other: Iterable[Any], /) -> SetMut[tuple[_K_co, _V_co]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rand__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __or__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __ror__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __sub__[T](self, other: Iterable[Any], /) -> SetMut[tuple[_K_co, _V_co]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rsub__[T](self, other: Iterable[T], /) -> SetMut[T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __xor__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def __rxor__[T](self, other: Iterable[T], /) -> SetMut[tuple[_K_co, _V_co] | T]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
    @override
    def intersection(self, other: Iterable[Any]) -> SetMut[tuple[_K_co, _V_co]]: ...
    @override
    def union[T](self, other: Iterable[T]) -> SetMut[tuple[_K_co, _V_co] | T]: ...
    @override
    def difference(self, other: Iterable[Any]) -> SetMut[tuple[_K_co, _V_co]]: ...
    @override
    def symmetric_difference[T](
        self, other: Iterable[T]
    ) -> SetMut[tuple[_K_co, _V_co] | T]: ...
