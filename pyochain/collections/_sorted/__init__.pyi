# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from typing import Any

from ._core import BaseSortedListSet
from ._dict import BaseSortedDict, SortedDict, SortedKeyDict
from ._keylist import SortedKeyList
from ._keyset import SortedKeySet
from ._list import BaseSortedList, SortedList
from ._set import BaseSortedSet, SortedSet
from ._views import SortedItemsView, SortedKeysView, SortedValuesView

def check_sorted_list(data: SortedList[Any]) -> None: ...
def check_sorted_set(data: BaseSortedSet[Any]) -> None: ...
def check_sorted_key_list(data: SortedKeyList[Any, Any]) -> None: ...
def assert_sorted_list_empty(lst: BaseSortedList[Any]) -> None: ...
def check_sorted_dict(data: BaseSortedDict[Any, Any]) -> None: ...

__all__ = [
    "BaseSortedDict",
    "BaseSortedList",
    "BaseSortedListSet",
    "BaseSortedSet",
    "SortedDict",
    "SortedItemsView",
    "SortedKeyDict",
    "SortedKeyList",
    "SortedKeySet",
    "SortedKeysView",
    "SortedList",
    "SortedSet",
    "SortedValuesView",
    "assert_sorted_list_empty",
    "check_sorted_dict",
    "check_sorted_key_list",
    "check_sorted_list",
    "check_sorted_set",
]
