from __future__ import annotations

from collections.abc import ItemsView, Iterable, KeysView, ValuesView
from typing import Any, Generic, TypeVar, override

from ._set import SetMut
from .abc import PyoMappingView, PyoSet

type AnyIter = Iterable[Any]
# TODO: It doesn't seem possible ATM to make Views generics work regarding covariance with the modern syntax.
V_co = TypeVar("V_co", covariant=True)
K_co = TypeVar("K_co", covariant=True)


class PyoValuesView[V](ValuesView[V], PyoMappingView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoKeysView(KeysView[K_co], PyoMappingView[K_co], PyoSet[K_co], Generic[K_co]):  # pyright: ignore[reportUnsafeMultipleInheritance]  # noqa: UP046
    """A view of the keys in a pyochain mapping.

    Keys views support set-like operations since dictionary keys are unique.

    See Also:
        `PyoMapping::keys`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AnyIter) -> SetMut[K_co]:
        return SetMut.from_ref(self & other)

    @override
    def union[T](self, other: Iterable[T]) -> SetMut[K_co | T]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AnyIter) -> SetMut[K_co]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference[T](self, other: Iterable[T]) -> SetMut[K_co | T]:
        return SetMut.from_ref(self ^ other)


class PyoItemsView(  # pyright: ignore[reportUnsafeMultipleInheritance]
    ItemsView[K_co, V_co],
    PyoMappingView[tuple[K_co, V_co]],
    PyoSet[tuple[K_co, V_co]],
    Generic[K_co, V_co],  # noqa: UP046
):
    """A view of the items (key-value pairs) in a pyochain mapping.

    Items are represented as tuples of `(key, value)` pairs, and the view supports set-like operations.

    See Also:
        `PyoMapping::items`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @override
    def intersection(self, other: AnyIter) -> SetMut[tuple[K_co, V_co]]:
        return SetMut.from_ref(self & other)

    @override
    def union[T](self, other: Iterable[T]) -> SetMut[tuple[K_co, V_co] | T]:
        return SetMut.from_ref(self | other)

    @override
    def difference(self, other: AnyIter) -> SetMut[tuple[K_co, V_co]]:
        return SetMut.from_ref(self - other)

    @override
    def symmetric_difference[T](
        self, other: Iterable[T]
    ) -> SetMut[tuple[K_co, V_co] | T]:
        return SetMut.from_ref(self ^ other)
