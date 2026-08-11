# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping, MutableMapping
from functools import partial
from typing import Any, Self, overload, override

from _typeshed import SupportsGetItem, SupportsKeysAndGetItem

from pyochain import Dict
from pyochain.abc import PyoIterator, PyoMutableMapping, PyoReversible
from pyochain.collections._sorted_views import (
    SortedItemsView,
    SortedKeysView,
    SortedValuesView,
    SupportsHashableAndRichComparison,
)
from pyochain.rs import SortedList

from ._core import SortedCollection

type KeyFunc[K: Hashable, OT: SupportsHashableAndRichComparison] = Callable[[K], OT]

class BaseSortedDict[K: SupportsHashableAndRichComparison, V](
    PyoMutableMapping[K, V], SortedCollection[K], PyoReversible[K], ABC
):
    _list: SortedList[K]
    _inner: Dict[K, V]

    @abstractmethod
    def __init__(
        self, iterable: Iterable[tuple[K, V]] | Mapping[K, V] = (), **kwargs: V
    ) -> None: ...
    @abstractmethod
    def __or__[T1, T2](self, value: Mapping[K, T2], /) -> SortedDict[K, V | T2]: ...
    @abstractmethod
    def __ror__[T1, T2](self, value: Mapping[K, T2], /) -> SortedDict[K, V | T2]: ...
    @override
    @abstractmethod
    def __reduce__(self) -> tuple[type[Self], tuple[Dict[K, V]]]:
        """Support for pickle.

        The tricks played with caching references in
        :func:`SortedDict.__init__` confuse pickle so customize the reducer.

        Returns:
            tuple[type[Self], tuple[Dict[K, V]]]: class and arguments for reconstruction

        """

    @property
    def inner(self) -> Dict[K, V]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __getitem__(self, key: K) -> V: ...
    @override
    def reset(self, load: int) -> None: ...
    @override
    def bisect_left(self, value: K) -> int: ...
    @override
    def bisect_right(self, value: K) -> int: ...
    @override
    def index(
        self, value: K, start: int | None = None, stop: int | None = None
    ) -> int: ...
    @override
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[K]: ...
    @override
    def irange(
        self,
        minimum: K | None = None,
        maximum: K | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[K]: ...
    @override
    def clear(self) -> None:
        """Remove all items from sorted dict.

        Runtime complexity: `O(n)`

        """

    @override
    def __delitem__(self, key: K) -> None:
        """Remove item from sorted dict identified by `key`.

        ``sd.__delitem__(key)`` <==> ``del sd[key]``

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            key (K): `key` for item lookup

        Examples:
            ```python
            import pytest
            from pyochain.collections import SortedDict

            sd = SortedDict({"a": 1, "b": 2, "c": 3})
            del sd["b"]
            assert sd == SortedDict({"a": 1, "c": 3})
            with pytest.raises(KeyError):
                del sd["z"]
            ```

        """
    @override
    def __iter__(self) -> PyoIterator[K]:
        """Return an iterator over the keys of the sorted dict.

        ``sd.__iter__()`` <==> ``iter(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """

    @override
    def __reversed__(self) -> PyoIterator[K]:
        """Return a reverse iterator over the keys of the sorted dict.

        ``sd.__reversed__()`` <==> ``reversed(sd)``

        Iterating the sorted dict while adding or deleting items may raise a
        :exc:`RuntimeError` or fail to iterate over all keys.

        """

    @override
    def __setitem__(self, key: K, value: V) -> None:
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

    def __ior__(
        self, other: Iterable[tuple[K, V]] | SupportsKeysAndGetItem[K, V]
    ) -> Self: ...
    @abstractmethod
    def copy(self) -> Self:
        """Return a shallow copy of the sorted dict.

        Runtime complexity: `O(n)`

        :return: new sorted dict

        """

    def __copy__(self) -> Self: ...
    @classmethod
    @overload
    def from_keys[OT: SupportsHashableAndRichComparison](
        cls, iterable: Iterable[OT], value: None = None, /
    ) -> SortedDict[OT, Any | None]: ...
    @classmethod
    @overload
    def from_keys[OT: SupportsHashableAndRichComparison, S](
        cls, iterable: Iterable[OT], value: S, /
    ) -> SortedDict[OT, S]: ...
    @classmethod
    def from_keys[OT: SupportsHashableAndRichComparison, S](
        cls, iterable: Iterable[OT], value: S | None = None, /
    ) -> SortedDict[OT, S | Any | None]:
        """Return a new sorted dict initialized from `iterable` and `value`.

        Items in the sorted dict have keys from `iterable` and values equal to
        `value`.

        Runtime complexity: `O(n*log(n))`

        :return: new sorted dict

        """

    @override
    def keys(self) -> SortedKeysView[K]:
        """Return new sorted keys view of the sorted dict's keys.

        See :class:`SortedKeysView` for details.

        :return: new sorted keys view

        """

    @override
    def items(self) -> SortedItemsView[K, V]:
        """Return new sorted items view of the sorted dict's items.

        See :class:`SortedItemsView` for details.

        :return: new sorted items view

        """

    @override
    def values(self) -> SortedValuesView[V]:
        """Return new sorted values view of the sorted dict's values.

        Note that the values view is sorted by key.

        See :class:`SortedValuesView` for details.

        :return: new sorted values view

        """

    @overload
    def pop(self, key: K, /) -> V: ...
    @overload
    def pop(self, key: K, /, default: V) -> V: ...
    @overload
    def pop[T](self, key: K, /, default: T) -> V | T: ...
    @override
    def pop[T](self, key: K, default: T = ...) -> V | T:
        """Remove and return value for item identified by `key`.

        If the `key` is not found then return `default` if given. If `default`
        is not given then raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            key (K): `key` for item
            default (T): `default` value if key not found (optional)

        Returns:
            V | T: value for item

        Examples:
            ```python
            from pyochain.collections import SortedDict
            import pytest

            sd = SortedDict({"a": 1, "b": 2, "c": 3})
            assert sd.pop("c") == 3
            assert sd.pop("z", 26) == 26
            with pytest.raises(KeyError):
                sd.pop("y")
            ```

        Raises:
            KeyError: if `key` not found and `default` not given

        """

    @override
    def popitem(self, index: int = -1) -> tuple[K, V]:
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

        Args:
            index (int): `index` of item (default -1)

        Returns:
            tuple[K, V]: key and value pair

        Raises:
            KeyError: if sorted dict is empty


        """

    def peekitem(self, index: int = -1) -> tuple[K, V]:
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

    @overload
    def setdefault[T](
        self: MutableMapping[K, T | None], key: K, default: None = None, /
    ) -> T | None: ...
    @overload
    def setdefault(self, key: K, default: V, /) -> V: ...
    @override
    def setdefault[T](self, key: K, default: V | T | None = None, /) -> V | T | None:
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
    @overload
    def update(self, m: SupportsKeysAndGetItem[K, V], /) -> None: ...
    @overload
    def update(
        self: SupportsGetItem[str, V], m: SupportsKeysAndGetItem[str, V], /, **kwargs: V
    ) -> None: ...
    @overload
    def update(self, m: Iterable[tuple[K, V]], /) -> None: ...
    @overload
    def update(
        self: SupportsGetItem[str, V], m: Iterable[tuple[str, V]], /, **kwargs: V
    ) -> None: ...
    @overload
    def update(self: SupportsGetItem[str, V], /, **kwargs: V) -> None: ...
    @override
    def update(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        m: SupportsKeysAndGetItem[K, V] | Iterable[tuple[K, V]] = (),
        /,
        **kwargs: V,
    ) -> None:
        """Update sorted dict with items from `args` and `kwargs`.

        Overwrites existing items.

        Optional arguments `args` and `kwargs` may be a mapping, an iterable of
        pairs or keyword arguments. See :func:`SortedDict.__init__` for
        details.

        :param args: mapping or iterable of pairs
        :param kwargs: keyword arguments mapping

        """

class SortedDict[K: SupportsHashableAndRichComparison, V](BaseSortedDict[K, V]):
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
    * :func:`SortedDict.from_keys`
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
    * :func:`SortedList.reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted dicts may only be compared for equality and inequality.


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

    def __init__(
        self, iterable: Iterable[tuple[K, V]] | Mapping[K, V] = (), **kwargs: V
    ) -> None: ...
    @override
    def __or__[T1, T2](self, value: Mapping[K, T2], /) -> SortedDict[K, V | T2]: ...
    @override
    def __ror__[T1, T2](self, value: Mapping[K, T2], /) -> SortedDict[K, V | T2]: ...
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[Dict[K, V]]]: ...
    @override
    def copy(self) -> Self: ...

class SortedKeyDict[
    K: SupportsHashableAndRichComparison,
    V,
    OT: SupportsHashableAndRichComparison,
](BaseSortedDict[K, V]):
    """Sorted dict with key-function for sorting keys.

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

    def __init__(
        self,
        iterable: Iterable[tuple[K, V]] | Mapping[K, V] = (),
        *,
        key: KeyFunc[K, OT],
        **kwargs: V,
    ) -> None: ...
    @property
    def key(self) -> KeyFunc[K, OT]:
        """Function used to extract comparison key from keys.

        Sorted dict compares keys directly when the key function is none.

        """

    @override
    def copy(self) -> Self: ...
    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[K]: ...
    def bisect_key_left(self, key: OT) -> int: ...
    def bisect_key_right(self, key: OT) -> int: ...
    @override
    def __ror__[T1, T2](  # pyright: ignore[reportIncompatibleMethodOverride]
        self, value: Mapping[K, T2], /
    ) -> SortedKeyDict[K, V | T2, OT]: ...
    @override
    def __or__[T1, T2](  # pyright: ignore[reportIncompatibleMethodOverride]
        self, value: Mapping[K, T2], /
    ) -> SortedKeyDict[K, V | T2, OT]: ...
    @override
    # pyrefly: ignore [bad-override]
    def __reduce__(self) -> tuple[partial[Self], tuple[Dict[K, V]]]: ...  # pyright: ignore[reportIncompatibleMethodOverride]
