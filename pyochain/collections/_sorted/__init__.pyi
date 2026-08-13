from typing import Any

from ._core import BaseSortedListSet
from ._dict import BaseSortedDict, SortedDict, SortedKeyDict
from ._list import BaseSortedList, SortedKeyList, SortedList
from ._set import BaseSortedSet, SortedKeySet, SortedSet
from ._views import SortedItemsView, SortedKeysView, SortedValuesView

def check_sorted_list(data: BaseSortedList[Any]) -> None: ...
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
