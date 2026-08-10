from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyochain.rs import check_sorted_list

if TYPE_CHECKING:
    from pyochain.collections import SortedDict


def check_sorted_dict(data: SortedDict[Any, Any]) -> None:
    list_ = data._list  # pyright: ignore[reportPrivateUsage]
    check_sorted_list(list_)
    assert data.len() == list_.len()
    assert list_.iter().all(data.contains)
