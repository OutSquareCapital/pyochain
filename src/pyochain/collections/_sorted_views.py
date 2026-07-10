# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from collections.abc import ItemsView, Iterable, KeysView, Sequence, ValuesView
from typing import overload, override

from ._sorted_set import SortedSet


class SortedKeysView[K](KeysView[K], Sequence[K]):
    """Sorted keys view is a dynamic view of the sorted dict's keys.

    When the sorted dict's keys change, the view reflects those changes.

    The keys view implements the set and sequence abstract base classes.

    """

    __slots__ = ()

    @classmethod
    @override
    def _from_iterable(cls, it: Iterable[K]) -> SortedSet[K]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return SortedSet(it)

    @overload
    def __getitem__(self, index: int) -> K: ...
    @overload
    def __getitem__(self, index: slice) -> list[K]: ...
    @override
    def __getitem__(self, index: int | slice) -> K | list[K]:
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
        ['a', 'b', 'c']
        >>> skv[100]
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param index: integer or slice for indexing

        Returns:
            key or list of keys
        :raises IndexError: if index out of range

        """
        return self._mapping._list[index]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


class SortedItemsView[K, V](ItemsView[K, V], Sequence[tuple[K, V]]):
    """Sorted items view is a dynamic view of the sorted dict's items.

    When the sorted dict's items change, the view reflects those changes.

    The items view implements the set and sequence abstract base classes.

    """

    __slots__ = ()

    @classmethod
    @override
    def _from_iterable(cls, it):
        return SortedSet(it)

    @overload
    def __getitem__(self, index: int) -> tuple[K, V]: ...
    @overload
    def __getitem__(self, index: slice) -> list[tuple[K, V]]: ...
    @override
    def __getitem__(self, index: int | slice) -> tuple[K, V] | list[tuple[K, V]]:
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

        :param index: integer or slice for indexing

        Returns:
            item or list of items

        """
        mapping = self._mapping
        mapping_list = mapping._list

        if isinstance(index, slice):
            keys = mapping_list[index]
            return [(key, mapping[key]) for key in keys]

        key = mapping_list[index]
        return key, mapping[key]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


class SortedValuesView[V](ValuesView[V], Sequence[V]):
    """Sorted values view is a dynamic view of the sorted dict's values.

    When the sorted dict's values change, the view reflects those changes.

    The values view implements the sequence abstract base class.

    """

    __slots__ = ()

    @overload
    def __getitem__(self, index: int) -> V: ...
    @overload
    def __getitem__(self, index: slice) -> list[V]: ...
    @override
    def __getitem__(self, index: int | slice) -> V | list[V]:
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

        :param index: integer or slice for indexing

        Returns:
            value or list of values
        """
        mapping = self._mapping
        mapping_list = mapping._list

        if isinstance(index, slice):
            keys = mapping_list[index]
            return [mapping[key] for key in keys]

        key = mapping_list[index]
        return mapping[key]

    def __delitem__(self, index: int | slice) -> None:
        return _view_delitem(self, index)


def _view_delitem(self, index) -> None:
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
    mapping = self._mapping
    list_ = mapping._list
    dict_delitem = dict.__delitem__
    if isinstance(index, slice):
        keys = list_[index]
        del list_[index]
        for key in keys:
            dict_delitem(mapping, key)
    else:
        key = list_.pop(index)
        dict_delitem(mapping, key)
