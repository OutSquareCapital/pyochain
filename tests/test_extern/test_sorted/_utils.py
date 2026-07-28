from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyochain.rs import check_sorted_list

if TYPE_CHECKING:
    from pyochain.collections import SortedDict, SortedSet
    from pyochain.rs import InnerSorted


def check_sorted_set(data: SortedSet[Any]) -> None:
    set_ = data._set  # pyright: ignore[reportPrivateUsage]
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_.inner)
    assert set_.len() == list_.len()
    assert list_.iter().all(set_.contains)


def check_sorted_dict(data: SortedDict[Any, Any]) -> None:
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_.inner)
    assert data.len() == list_.len()
    assert list_.iter().all(data.contains)


def assert_sorted_list_empty(lst: InnerSorted[Any, Any]) -> None:
    assert lst.len == 0
    assert lst.maxes == []
    assert lst.lists == []
