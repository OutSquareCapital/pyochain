# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from collections.abc import Iterator, Mapping
from itertools import chain
from reprlib import recursive_repr
from typing import TYPE_CHECKING, Self, override

from ._sorted_list import KeyFunc, SortedKeyList, SortedList
from ._sorted_views import SortedItemsView, SortedKeysView, SortedValuesView

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


class SortedDict[K, V](dict[K, V]):  # noqa: FURB189
    """Sorted dict is a sorted mutable mapping.

    Sorted dict keys are maintained in sorted order. The design of sorted dict
    is simple: sorted dict inherits from dict to store items and maintains a
    sorted list of keys.

    Sorted dict keys must be hashable and comparable. The hash and total
    ordering of keys must not change while they are stored in the sorted dict.

    Mutable mapping methods:

    * :func:`SortedDict.__getitem__` (inherited from dict)
    * :func:`SortedDict.__setitem__`
    * :func:`SortedDict.__delitem__`
    * :func:`SortedDict.__iter__`
    * :func:`SortedDict.__len__` (inherited from dict)

    Methods for adding items:

    * :func:`SortedDict.setdefault`
    * :func:`SortedDict.update`

    Methods for removing items:

    * :func:`SortedDict.clear`
    * :func:`SortedDict.pop`
    * :func:`SortedDict.popitem`

    Methods for looking up items:

    * :func:`SortedDict.__contains__` (inherited from dict)
    * :func:`SortedDict.get` (inherited from dict)
    * :func:`SortedDict.peekitem`

    Methods for views:

    * :func:`SortedDict.keys`
    * :func:`SortedDict.items`
    * :func:`SortedDict.values`

    Methods for miscellany:

    * :func:`SortedDict.copy`
    * :func:`SortedDict.fromkeys`
    * :func:`SortedDict.__reversed__`
    * :func:`SortedDict.__eq__` (inherited from dict)
    * :func:`SortedDict.__ne__` (inherited from dict)
    * :func:`SortedDict.__repr__`
    * :func:`SortedDict._check`

    Sorted list methods available (applies to keys):

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList._reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted dicts may only be compared for equality and inequality.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize sorted dict instance.

        Optional key-function argument defines a callable that, like the `key`
        argument to the built-in `sorted` function, extracts a comparison key
        from each dictionary key. If no function is specified, the default
        compares the dictionary keys directly. The key-function argument must
        be provided as a positional argument and must come before all other
        arguments.

        Optional iterable argument provides an initial sequence of pairs to
        initialize the sorted dict. Each pair in the sequence defines the key
        and corresponding value. If a key is seen more than once, the last
        value associated with it is stored in the new sorted dict.

        Optional mapping argument provides an initial mapping of items to
        initialize the sorted dict.

        If keyword arguments are given, the keywords themselves, with their
        associated values, are added as items to the dictionary. If a key is
        specified both in the positional argument and as a keyword argument,
        the value associated with the keyword is stored in the
        sorted dict.

        Sorted dict keys must be hashable, per the requirement for Python's
        dictionaries. Keys (or the result of the key-function) must also be
        comparable, per the requirement for sorted lists.

        >>> d = {"alpha": 1, "beta": 2}
        >>> SortedDict([("alpha", 1), ("beta", 2)]) == d
        True
        >>> SortedDict({"alpha": 1, "beta": 2}) == d
        True
        >>> SortedDict(alpha=1, beta=2) == d
        True

        """
        if args and (args[0] is None or callable(args[0])):
            args = args[1:]

            self._list = SortedList()
        else:
            self._list = SortedList()

        # Reaching through ``self._list`` repeatedly adds unnecessary overhead
        # so cache references to sorted list methods.

        list_ = self._list
        self._list_add = list_.add
        self._list_clear = list_.clear
        self._list_iter = list_.__iter__
        self._list_reversed = list_.__reversed__
        self._list_pop = list_.pop
        self._list_remove = list_.remove
        self._list_update = list_.update

        # Expose some sorted list methods publicly.

        self.bisect_left = list_.bisect_left
        self.bisect = list_.bisect_right
        self.bisect_right = list_.bisect_right
        self.index = list_.index
        self.irange = list_.irange
        self.islice = list_.islice
        self._reset = list_._reset

        self._update(*args, **kwargs)

    @override
    def clear(self) -> None:
        """Remove all items from sorted dict.

        Runtime complexity: `O(n)`

        """
        dict.clear(self)
        self._list_clear()

    @override
    def __delitem__(self, key) -> None:
        """Remove item from sorted dict identified by `key`.

        ``sd.__delitem__(key)`` <==> ``del sd[key]``

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> del sd["b"]
        >>> sd
        SortedDict({'a': 1, 'c': 3})
        >>> del sd["z"]
        Traceback (most recent call last):
          ...
        KeyError: 'z'

        :param key: `key` for item lookup
        :raises KeyError: if key not found

        """
        dict.__delitem__(self, key)
        self._list_remove(key)

    @override
    def __iter__(self):
        """Return an iterator over the keys of the sorted dict.

        ``sd.__iter__()`` <==> ``iter(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """
        return self._list_iter()

    @override
    def __reversed__(self):
        """Return a reverse iterator over the keys of the sorted dict.

        ``sd.__reversed__()`` <==> ``reversed(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """
        return self._list_reversed()

    @override
    def __setitem__(self, key, value) -> None:
        """Store item in sorted dict with `key` and corresponding `value`.

        ``sd.__setitem__(key, value)`` <==> ``sd[key] = value``

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict()
        >>> sd["c"] = 3
        >>> sd["a"] = 1
        >>> sd["b"] = 2
        >>> sd
        SortedDict({'a': 1, 'b': 2, 'c': 3})

        :param key: key for item
        :param value: value for item

        """
        if key not in self:
            self._list_add(key)
        dict.__setitem__(self, key, value)

    _setitem = __setitem__

    @override
    def __or__(self, other: object):
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(self.items(), other.items())
        return self.__class__(items)

    @override
    def __ror__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(other.items(), self.items())
        return self.__class__(items)

    @override
    def __ior__(self, other):
        self._update(other)
        return self

    @override
    def copy(self) -> Self:
        """Return a shallow copy of the sorted dict.

        Runtime complexity: `O(n)`

        :return: new sorted dict

        """
        return self.__class__(self.items())

    __copy__ = copy

    @classmethod
    @override
    def fromkeys(cls, iterable, value=None):
        """Return a new sorted dict initailized from `iterable` and `value`.

        Items in the sorted dict have keys from `iterable` and values equal to
        `value`.

        Runtime complexity: `O(n*log(n))`

        :return: new sorted dict

        """
        return cls((key, value) for key in iterable)

    @override
    def keys(self):
        """Return new sorted keys view of the sorted dict's keys.

        See :class:`SortedKeysView` for details.

        :return: new sorted keys view

        """
        return SortedKeysView(self)

    @override
    def items(self):
        """Return new sorted items view of the sorted dict's items.

        See :class:`SortedItemsView` for details.

        :return: new sorted items view

        """
        return SortedItemsView(self)

    @override
    def values(self):
        """Return new sorted values view of the sorted dict's values.

        Note that the values view is sorted by key.

        See :class:`SortedValuesView` for details.

        :return: new sorted values view

        """
        return SortedValuesView(self)

    class _NotGiven:
        # pylint: disable=too-few-public-methods
        @override
        def __repr__(self) -> str:
            return "<not-given>"

    __not_given = _NotGiven()

    @override
    def pop(self, key, default=__not_given):
        """Remove and return value for item identified by `key`.

        If the `key` is not found then return `default` if given. If `default`
        is not given then raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> sd.pop("c")
        3
        >>> sd.pop("z", 26)
        26
        >>> sd.pop("y")
        Traceback (most recent call last):
          ...
        KeyError: 'y'

        :param key: `key` for item
        :param default: `default` value if key not found (optional)

        Raises:
            KeyError: if `key` not found and `default` not given

        Returns:
            value: value for item

        """
        if key in self:
            self._list_remove(key)
            return dict.pop(self, key)
        if default is self.__not_given:
            raise KeyError(key)
        return default

    @override
    def popitem(self, index=-1):
        """Remove and return ``(key, value)`` pair at `index` from sorted dict.

        Optional argument `index` defaults to -1, the last item in the sorted
        dict. Specify ``index=0`` for the first item in the sorted dict.

        If the sorted dict is empty, raises :exc:`KeyError`.

        If the `index` is out of range, raises :exc:`IndexError`.

        Runtime complexity: `O(log(n))`

        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> sd.popitem()
        ('c', 3)
        >>> sd.popitem(0)
        ('a', 1)
        >>> sd.popitem(100)
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param int index: `index` of item (default -1)

        Returns:
            tuple: key and value pair

        Raises:
            KeyError: if sorted dict is empty


        """
        if not self:
            msg = "popitem(): dictionary is empty"
            raise KeyError(msg)

        key = self._list_pop(index)
        value = dict.pop(self, key)
        return (key, value)

    def peekitem(self, index=-1):
        """Return ``(key, value)`` pair at `index` in sorted dict.

        Optional argument `index` defaults to -1, the last item in the sorted
        dict. Specify ``index=0`` for the first item in the sorted dict.

        Unlike :func:`SortedDict.popitem`, the sorted dict is not modified.

        If the `index` is out of range, raises :exc:`IndexError`.

        Runtime complexity: `O(log(n))`

        >>> sd = SortedDict({"a": 1, "b": 2, "c": 3})
        >>> sd.peekitem()
        ('c', 3)
        >>> sd.peekitem(0)
        ('a', 1)
        >>> sd.peekitem(100)
        Traceback (most recent call last):
          ...
        IndexError: list index out of range

        :param int index: index of item (default -1)
        :return: key and value pair
        :raises IndexError: if `index` out of range

        """
        key = self._list[index]
        return key, self[key]

    @override
    def setdefault(self, key, default=None):
        """Return value for item identified by `key` in sorted dict.

        If `key` is in the sorted dict then return its value. If `key` is not
        in the sorted dict then insert `key` with value `default` and return
        `default`.

        Optional argument `default` defaults to none.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sd = SortedDict()
        >>> sd.setdefault("a", 1)
        1
        >>> sd.setdefault("a", 10)
        1
        >>> sd
        SortedDict({'a': 1})

        :param key: key for item
        :param default: value for item (default None)
        :return: value for item identified by `key`

        """
        if key in self:
            return self[key]
        dict.__setitem__(self, key, default)
        self._list_add(key)
        return default

    @override
    def update(self, *args, **kwargs) -> None:
        """Update sorted dict with items from `args` and `kwargs`.

        Overwrites existing items.

        Optional arguments `args` and `kwargs` may be a mapping, an iterable of
        pairs or keyword arguments. See :func:`SortedDict.__init__` for
        details.

        :param args: mapping or iterable of pairs
        :param kwargs: keyword arguments mapping

        """
        if not self:
            dict.update(self, *args, **kwargs)
            self._list_update(dict.__iter__(self))
            return

        if not kwargs and len(args) == 1 and isinstance(args[0], dict):
            pairs = args[0]
        else:
            pairs = dict(*args, **kwargs)

        if (10 * len(pairs)) > len(self):
            dict.update(self, pairs)
            self._list_clear()
            self._list_update(dict.__iter__(self))
        else:
            for key in pairs:
                self._setitem(key, pairs[key])

    _update = update

    @override
    def __reduce__(self):
        """Support for pickle.

        The tricks played with caching references in
        :func:`SortedDict.__init__` confuse pickle so customize the reducer.

        Returns:
            tuple: class and arguments for reconstruction

        """
        items = dict.copy(self)
        return (type(self), (items,))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted dict.

        ``sd.__repr__()`` <==> ``repr(sd)``

        :return: string representation

        """
        type_name = type(self).__name__
        item_format = "{!r}: {!r}".format
        items = ", ".join(item_format(key, self[key]) for key in self._list)
        return f"{type_name}({{{items}}})"

    def _check(self) -> None:
        """Check invariants of sorted dict.

        Runtime complexity: `O(n)`

        """
        list_ = self._list
        list_._check()
        assert len(self) == len(list_)
        assert all(key in self for key in list_)


class SortedKeyDict[K, V, OT: SupportsRichComparison](SortedDict[K, V]):
    def __init__(self, *args, **kwargs) -> None:
        """Initialize sorted dict instance.

        Optional key-function argument defines a callable that, like the `key`
        argument to the built-in `sorted` function, extracts a comparison key
        from each dictionary key. If no function is specified, the default
        compares the dictionary keys directly. The key-function argument must
        be provided as a positional argument and must come before all other
        arguments.

        Optional iterable argument provides an initial sequence of pairs to
        initialize the sorted dict. Each pair in the sequence defines the key
        and corresponding value. If a key is seen more than once, the last
        value associated with it is stored in the new sorted dict.

        Optional mapping argument provides an initial mapping of items to
        initialize the sorted dict.

        If keyword arguments are given, the keywords themselves, with their
        associated values, are added as items to the dictionary. If a key is
        specified both in the positional argument and as a keyword argument,
        the value associated with the keyword is stored in the
        sorted dict.

        Sorted dict keys must be hashable, per the requirement for Python's
        dictionaries. Keys (or the result of the key-function) must also be
        comparable, per the requirement for sorted lists.

        >>> d = {"alpha": 1, "beta": 2}
        >>> SortedDict([("alpha", 1), ("beta", 2)]) == d
        True
        >>> SortedDict({"alpha": 1, "beta": 2}) == d
        True
        >>> SortedDict(alpha=1, beta=2) == d
        True

        """
        self._key: KeyFunc[K, OT] = args[0]
        args = args[1:]

        self._list = SortedKeyList(key=self._key)

        # Reaching through ``self._list`` repeatedly adds unnecessary overhead
        # so cache references to sorted list methods.

        list_ = self._list
        self._list_add = list_.add
        self._list_clear = list_.clear
        self._list_iter = list_.__iter__
        self._list_reversed = list_.__reversed__
        self._list_pop = list_.pop
        self._list_remove = list_.remove
        self._list_update = list_.update

        # Expose some sorted list methods publicly.

        self.bisect_left = list_.bisect_left
        self.bisect = list_.bisect_right
        self.bisect_right = list_.bisect_right
        self.index = list_.index
        self.irange = list_.irange
        self.islice = list_.islice
        self._reset = list_._reset

        self._update(*args, **kwargs)

    @property
    def key(self) -> KeyFunc[K, OT]:
        """Function used to extract comparison key from keys.

        Sorted dict compares keys directly when the key function is none.

        """
        return self._key

    @override
    def copy(self) -> Self:
        return self.__class__(self._key, self.items())

    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        reverse: bool = False,
    ) -> Iterator[K]:
        return self._list.irange_key(min_key, max_key, inclusive, reverse)

    def bisect_key_left(self, key: OT) -> int:
        return self._list.bisect_key_left(key)

    def bisect_key_right(self, key: OT) -> int:
        return self._list.bisect_key_right(key)

    def bisect_key(self, key: OT) -> int:
        return self._list.bisect_key(key)

    @override
    def __ror__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(other.items(), self.items())
        return self.__class__(self._key, items)

    @override
    def __or__(self, other: object) -> Self:
        if not isinstance(other, Mapping):
            return NotImplemented
        items = chain(self.items(), other.items())
        return self.__class__(self._key, items)

    @override
    def __reduce__(self):
        items = dict.copy(self)
        return (type(self), (self._key, items))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted dict.

        ``sd.__repr__()`` <==> ``repr(sd)``

        :return: string representation

        """
        key = self._key
        type_name = type(self).__name__
        key_arg = "" if key is None else f"{key!r}, "
        item_format = "{!r}: {!r}".format
        items = ", ".join(item_format(key, self[key]) for key in self._list)
        return f"{type_name}({key_arg}{{{items}}})"
