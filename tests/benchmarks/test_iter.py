from collections.abc import Callable

import pytest

from pyochain import Range, Seq

type BenchFn = Callable[[int], object]
type BenchFixture = Callable[[BenchFn, int], object]


def _pyochain_filter_map(size: int) -> Seq[int]:
    return (
        Range(0, size)
        .iter()
        .map(lambda x: x * 2)
        .filter(lambda x: x % 3 == 0)
        .collect()
    )


def _python_filter_map(size: int) -> tuple[int, ...]:
    return tuple(
        filter(
            lambda x: x % 3 == 0,
            map(lambda x: x * 2, range(size)),
        )
    )


@pytest.mark.benchmark(group="filter_map")
@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(_python_filter_map, id="python"),
        pytest.param(_pyochain_filter_map, id="pyochain"),
    ],
)
@pytest.mark.parametrize("size", [10, 100, 500])
def test_filter_map(benchmark: BenchFixture, fn: BenchFn, size: int) -> None:
    result = benchmark(fn, size)
    assert result
