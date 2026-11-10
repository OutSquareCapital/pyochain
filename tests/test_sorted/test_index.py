from __future__ import annotations

import pytest

from ._utils import LIST_CLASSES, LOAD, List, reset_list


@LIST_CLASSES
def test_stop_below_neg_len[T: List](cls: type[T]) -> None:
    """Should raise `ValueError` if stop is below -len(list)."""
    sl = cls(range(LOAD))
    reset_list(sl)
    with pytest.raises(ValueError):
        _ = sl.index(0, 0, -LOAD - 1)
