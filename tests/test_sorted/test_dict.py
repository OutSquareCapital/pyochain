from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sortedcontainers import SortedDict as PySorteDict

from pyochain import Iter
from pyochain.collections import SortedDict

if TYPE_CHECKING:
    from ._utils import Dict


@pytest.mark.parametrize(
    "cls",
    (
        pytest.param(dict),
        pytest.param(SortedDict, id="pyochain"),
        pytest.param(PySorteDict, id="sortedcontainers"),
    ),
)
def test_del_and_iter(cls: type[Dict]) -> None:
    """`sortedcontainers` issue # 141.

    https://github.com/grantjenks/python-sortedcontainers/issues/141"""
    match Iter.from_count(1).map(lambda i: (i, 0)).take(4).collect(cls):
        case SortedDict() | PySorteDict() as sortedict:
            _iter_del(sortedict)
            assert tuple(sortedict) == (2, 4)
        case pydict:
            with pytest.raises(RuntimeError):
                _iter_del(pydict)


def _iter_del(d: Dict) -> None:
    for k in d:
        del d[k]  # ruff: ignore[loop-iterator-mutation]
