# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload, override

from pyochain._types import SupportsHashableAndRichComparison
from pyochain.abc import PyoItemsView, PyoKeysView, PyoSequence, PyoValuesView

from ._sorted_set import SortedSet

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pyochain import Vec
    from pyochain.collections import SortedDict

_K_co = TypeVar("_K_co", covariant=True, bound=SupportsHashableAndRichComparison)
_V_co = TypeVar("_V_co", covariant=True)


class SortedKeysView(PyoKeysView[_K_co], PyoSequence[_K_co], Generic[_K_co]):  # ruff:ignore[non-pep695-generic-class]
    """Sorted keys view is a dynamic view of the sorted dict's keys.

    When the sorted dict's keys change, the view reflects those changes.

    The keys view implements the set and sequence abstract base classes.

    """

    _mapping: SortedDict[_K_co, Any]  # pyright: ignore[reportIncompatibleVariableOverride]

    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @classmethod
    @override
    def _from_iterable(cls, it: Iterable[_K_co]) -> SortedSet[_K_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return SortedSet(it)

    @overload
    def __getitem__(self, index: int) -> _K_co: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[_K_co]: ...
    @override
    def __getitem__(self, index: int | slice) -> _K_co | Vec[_K_co]:
        """Lookup key at `index` in sorted keys views.

        ``skv.__getitem__(index)`` <==> ``skv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedDict
        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> skv = sd.keys()
        >>> skv[0]
        'a'
        >>> skv[-1]
        'c'
        >>> skv[:]
        Vec('a', 'b', 'c')
        >>> skv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            _K_co | Vec[_K_co]: key or list of keys

        :raises IndexError: if index out of range

        """
        return self._mapping._list[index]  # pyright: ignore[reportPrivateUsage]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


class SortedItemsView(
    PyoItemsView[_K_co, _V_co],
    PyoSequence[tuple[_K_co, _V_co]],
    Generic[_K_co, _V_co],  # ruff:ignore[non-pep695-generic-class]
):
    """Sorted items view is a dynamic view of the sorted dict's items.

    When the sorted dict's items change, the view reflects those changes.

    The items view implements the set and sequence abstract base classes.

    """

    _mapping: SortedDict[_K_co, _V_co]  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @classmethod
    @override
    def _from_iterable(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, it: Iterable[tuple[_K_co, _V_co]]
    ) -> SortedSet[tuple[_K_co, _V_co]]:
        return SortedSet(it)

    @overload
    def __getitem__(self, index: int) -> tuple[_K_co, _V_co]: ...
    @overload
    def __getitem__(self, index: slice) -> list[tuple[_K_co, _V_co]]: ...
    @override
    def __getitem__(
        self, index: int | slice
    ) -> tuple[_K_co, _V_co] | list[tuple[_K_co, _V_co]]:
        """Lookup item at `index` in sorted items view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedDict
        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> siv = sd.items()
        >>> siv[0]
        ('a', 1)
        >>> siv[-1]
        ('c', 3)
        >>> siv[:]
        [('a', 1), ('b', 2), ('c', 3)]
        >>> siv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            tuple[_K_co, _V_co] | list[tuple[_K_co, _V_co]]: item or list of items

        """
        mapping = self._mapping
        mapping_list = mapping._list  # pyright: ignore[reportPrivateUsage]

        match index:
            case slice():
                keys = mapping_list[index]
                return [(key, mapping[key]) for key in keys]
            case int():
                key = mapping_list[index]
                return key, mapping[key]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


class SortedValuesView(PyoValuesView[_V_co], PyoSequence[_V_co], Generic[_V_co]):  # ruff:ignore[non-pep695-generic-class]
    """Sorted values view is a dynamic view of the sorted dict's values.

    When the sorted dict's values change, the view reflects those changes.

    The values view implements the sequence abstract base class.

    """

    _mapping: SortedDict[Any, _V_co]  # pyright: ignore[reportIncompatibleVariableOverride]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute, reportIncompatibleUnannotatedOverride]

    @overload
    def __getitem__(self, index: int) -> _V_co: ...
    @overload
    def __getitem__(self, index: slice) -> list[_V_co]: ...
    @override
    def __getitem__(self, index: int | slice) -> _V_co | list[_V_co]:
        """Lookup value at `index` in sorted values view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from pyochain.collections import SortedDict
        >>> sd = SortedDict({"a": 2, "b": 1, "c": 3})
        >>> svv = sd.values()
        >>> svv[0]
        2
        >>> svv[-1]
        3
        >>> svv[:]
        [2, 1, 3]
        >>> svv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            _V_co | list[_V_co]: value or list of values
        """
        mapping = self._mapping
        mapping_list = mapping._list  # pyright: ignore[reportPrivateUsage]

        match index:
            case slice():
                return [mapping[key] for key in mapping_list[index]]  # pyright: ignore[reportAny]
            case int():
                return mapping[mapping_list[index]]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


def _view_delitem[K: SupportsHashableAndRichComparison, V](
    self: SortedKeysView[K] | SortedValuesView[V] | SortedItemsView[K, V],
    index: int | slice,
) -> None:
    """Remove item at `index` from sorted dict.

    ``view.__delitem__(index)`` <==> ``del view[index]``

    Supports slicing.

    Runtime complexity: `O(log(n))` -- approximate.

    >>> from pyochain.collections import SortedDict
    >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
    >>> view = sd.keys()
    >>> del view[0]
    >>> sd
    SortedDict({'b': 2, 'c': 3})
    >>> del view[-1]
    >>> sd
    SortedDict({'b': 2})
    >>> del view[:]
    >>> sd
    SortedDict({})

    :param index: integer or slice for indexing
    :raises IndexError: if index out of range

    """
    mapping = self._mapping  # pyright: ignore[reportPrivateUsage]
    list_ = mapping._list  # pyright: ignore[reportPrivateUsage]
    dict_delitem = dict[K, V].__delitem__
    match index:
        case slice():
            keys = list_[index]
            del list_[index]
            for key in keys:
                dict_delitem(mapping, key)
        case int():
            key = list_.pop(index)
            dict_delitem(mapping, key)
