from __future__ import annotations

from collections.abc import ItemsView, Iterable, KeysView, ValuesView
from typing import Any, override

from ._set import SetMut
from .abc import PyoCollection, PyoMappingView, PyoSet

type AnyIter = Iterable[Any]


class PyoValuesView[V](PyoMappingView, PyoCollection[V], ValuesView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoKeysView[K](PyoMappingView, PyoSet[K], KeysView[K]):  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]
    """A view of the keys in a pyochain mapping.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AnyIter) -> SetMut[K]:
        return SetMut.from_ref(self & other)

    @override
    def union[S, T](self: PyoKeysView[S], other: Iterable[T]) -> SetMut[S | T]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AnyIter) -> SetMut[K]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference[S, T](
        self: PyoKeysView[S], other: Iterable[T]
    ) -> SetMut[S | T]:
        return SetMut.from_ref(self ^ other)


class PyoItemsView[K, V](  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]
    PyoMappingView,
    PyoSet[tuple[K, V]],
    ItemsView[K, V],
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping::items`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AnyIter) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self & other)

    @override
    def union[T](self, other: Iterable[T]) -> SetMut[tuple[K, V] | T]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AnyIter) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference[T](self, other: Iterable[T]) -> SetMut[tuple[K, V] | T]:
        return SetMut.from_ref(self ^ other)
