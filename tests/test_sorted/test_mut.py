from __future__ import annotations

import pytest
import sortedcontainers as sc

from ._utils import LIST_CLASSES, List


@LIST_CLASSES
def test_mut_empty_collection(cls: type[List]) -> None:
    x = cls()
    # On a slice, it's a no-op
    del x[0:1]
    # On an index, it raises an IndexError
    with pytest.raises(IndexError):
        del x[-1]
    match x:
        case sc.SortedList():
            _try_setitem(x, NotImplementedError)
        case _:
            _try_setitem(x, IndexError)


def _try_setitem(x: List, err: type[Exception]) -> None:
    with pytest.raises(err):
        x[-1] = 0
