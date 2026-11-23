"""Additional benchmarks not presents in the original `sortedcontainers` package."""

import pytest

from benchmarks._utils import BenchFixture

from ._utils import LST_PARAMS, List, shuffled, update


@LST_PARAMS
@pytest.mark.parametrize("size", (1_000, 100_000, 1_000_000))
def test_update_density(
    benchmark: BenchFixture, cls: type[List], size: int, per_sublist: int
) -> None:
    """Extends with `per_sublist` values for each sublist of the default load (1000)."""
    update(benchmark, cls, size, shuffled(size, 1_000 // per_sublist))
