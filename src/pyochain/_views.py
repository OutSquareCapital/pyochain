from __future__ import annotations

from collections.abc import ItemsView, Iterable, KeysView, ValuesView
from typing import Any, Generic, TypeVar, override

from ._set import SetMut
from .abc import PyoCollection, PyoMappingView, PyoSet

type AnyIter = Iterable[Any]
# TODO: It doesn't seem possible ATM to make Views generics work regarding covariance with the modern syntax.
V_co = TypeVar("V_co", covariant=True)
K_co = TypeVar("K_co", covariant=True)


class PyoValuesView[V](PyoMappingView, PyoCollection[Any], ValuesView[V]):  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]
    """A view of the values in a pyochain mapping.

    See Also:
        `PyoMapping::values`: Method that returns this view.
    """

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]


class PyoKeysView(PyoMappingView, PyoSet[K_co], KeysView[K_co], Generic[K_co]):  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]  # noqa: UP046
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


class PyoItemsView(  # pyright: ignore[reportUnsafeMultipleInheritance, reportImplicitAbstractClass]
    PyoMappingView,
    PyoSet[tuple[K_co, V_co]],
    ItemsView[K_co, V_co],
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
