from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._utils import DICT_PARAMS, SC_SIZES, Dict, run_pedantic, shuffled_pairs

if TYPE_CHECKING:
    from .._utils import BenchFixture


@pytest.mark.parametrize(
    "empty", (pytest.param(False, id="filled"), pytest.param(True, id="empty"))
)
@DICT_PARAMS
@SC_SIZES
def test_setitem(
    benchmark: BenchFixture, cls: type[Dict], size: int, empty: bool
) -> None:

    values = shuffled_pairs(size, 100)

    def f(dct: Dict) -> None:
        values.iter().for_each_star(dct.__setitem__)

    run_pedantic(benchmark, cls, size, f, empty=empty)
