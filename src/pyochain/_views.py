from __future__ import annotations

from collections.abc import ItemsView, KeysView, ValuesView
from collections.abc import Set as AbstractSet
from typing import override

from ._set import BaseConcreteSet, SetMut
from .abc import PyoMappingView, PyoSet


class PyoValuesView[V](ValuesView[V], PyoMappingView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoKeysView[K](KeysView[K], PyoMappingView[K], PyoSet[K], BaseConcreteSet[K]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the keys in a pyochain mapping.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self & other)

    @override
    def union(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference(self, other: AbstractSet[K]) -> SetMut[K]:
        return SetMut.from_ref(self ^ other)


class PyoItemsView[K, V](  # pyright: ignore[reportUnsafeMultipleInheritance]
    ItemsView[K, V],
    PyoMappingView[tuple[K, V]],
    PyoSet[tuple[K, V]],
    BaseConcreteSet[tuple[K, V]],
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping::items`: Method that returns this view.
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self & other)

    @override
    def union(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AbstractSet[tuple[K, V]]) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference(
        self, other: AbstractSet[tuple[K, V]]
    ) -> SetMut[tuple[K, V]]:
        return SetMut.from_ref(self ^ other)
