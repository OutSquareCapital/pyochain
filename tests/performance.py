import statistics
import time
from collections.abc import Callable, Sequence

import pyochain as pc

type Tested = Callable[[], Sequence[Sequence[int]]]

INNER = range(1000)
OUTER = range(100)


def _measure(func: Tested, iterations: int) -> float:
    all_time: list[float] = []
    for _ in range(iterations):
        start_time = time.perf_counter()
        func()
        all_time.append(time.perf_counter() - start_time)
    return statistics.median(all_time)


def test_performance_iter_map(iterations: int) -> None:
    def _add_measure(data: pc.Dict[str, float], func: Tested) -> pc.Dict[str, float]:
        return data.with_key(func.__name__, _measure(func, iterations))

    def square(x: int) -> int:
        return x * x

    def _iter_map() -> Sequence[Sequence[int]]:
        return (
            pc.Seq(OUTER)
            .iter()
            .map(lambda _: pc.Seq(INNER).iter().map(square).collect().inner())
            .collect()
            .inner()
        )

    def _built_in_map() -> Sequence[Sequence[int]]:
        return list(map(lambda _: list(map(square, INNER)), OUTER))

    def _for_loop() -> Sequence[Sequence[int]]:
        total: list[list[int]] = []
        for _ in OUTER:
            result: list[int] = []
            for x in INNER:
                result.append(square(x))
            total.append(result)
        return total

    def _comprehension() -> Sequence[Sequence[int]]:
        return [[square(x) for x in INNER] for _ in OUTER]

    def _assert_equals():
        assert _iter_map() == _built_in_map() == _for_loop() == _comprehension()
        print("okay, starting performance test...")

    def _run_test():
        return (
            pc.Dict[str, float]({})
            .pipe(_add_measure, _iter_map)
            .pipe(_add_measure, _built_in_map)
            .pipe(_add_measure, _for_loop)
            .pipe(_add_measure, _comprehension)
            .sort_values()
            .for_each(lambda k, v: print(f"{k}: {v:.6f} seconds"))
        )

    _assert_equals()
    _run_test()
