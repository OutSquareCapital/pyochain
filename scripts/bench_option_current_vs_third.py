from __future__ import annotations

import statistics
import timeit
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from rich.console import Console
from rich.table import Table

import pyochain as pc
import type_approaches as new_pc


def _quartiles_25_75(values: pc.Seq[float]) -> tuple[float, float]:
    """Calculate 25th and 75th percentiles."""
    sorted_vals = values.iter().sort()
    n = sorted_vals.length()
    if n < 2:
        return sorted_vals.first(), sorted_vals.first()

    q1_idx = int((n - 1) * 0.25)
    q3_idx = int((n - 1) * 0.75)
    return sorted_vals[q1_idx], sorted_vals[q3_idx]


class OptionProto[T](Protocol):
    """Protocol for Option-like types used in benchmarks."""

    def map(self, func: Callable[[Any], object]) -> OptionProto[Any]: ...

    def filter(self, predicate: Callable[[Any], bool]) -> OptionProto[Any]: ...

    def and_then(self, func: Callable[[Any], OptionProto[Any]]) -> OptionProto[Any]: ...

    def zip(self, other: OptionProto[Any]) -> OptionProto[Any]: ...
    def unwrap_or(self, default: T) -> T: ...


class ResultProto[T, E](Protocol):
    """Protocol for Result-like types used in benchmarks."""

    def is_ok(self) -> bool: ...

    def is_err(self) -> bool: ...

    def unwrap(self) -> T: ...

    def unwrap_err(self) -> E: ...

    def map(self, func: Callable[[Any], object]) -> ResultProto[Any, Any]: ...

    def map_err(self, func: Callable[[Any], object]) -> ResultProto[Any, Any]: ...

    def map_or_else(
        self, ok: Callable[[Any], Any], err: Callable[[Any], Any]
    ) -> Any: ...

    def and_then(
        self, func: Callable[[Any], ResultProto[Any, Any]]
    ) -> ResultProto[Any, Any]: ...

    def or_else(
        self, func: Callable[[Any], ResultProto[Any, Any]]
    ) -> ResultProto[Any, Any]: ...

    def is_ok_and(self, pred: Callable[[Any], bool]) -> bool: ...

    def is_err_and(self, pred: Callable[[Any], bool]) -> bool: ...

    def unwrap_or(self, default: T) -> T: ...


@dataclass(frozen=True, slots=True)
class Case:
    """One benchmark case (current vs new_pc) with multiple runs."""

    name: str
    current_times: pc.Seq[float]  # Multiple runs for current
    new_pc_times: pc.Seq[float]  # Multiple runs for new_pc

    @property
    def current_median(self) -> float:
        """Median time for current implementation."""
        return statistics.median(self.current_times)

    @property
    def new_pc_median(self) -> float:
        """Median time for new_pc implementation."""
        return statistics.median(self.new_pc_times)

    @property
    def _improvements(self) -> pc.Seq[float]:
        """Improvements across all runs (cached)."""
        return (
            self.current_times.iter()
            .zip(self.new_pc_times, strict=True)
            .map(lambda pair: ((pair[1] - pair[0]) / pair[1]) * 100)
            .collect()
        )

    @property
    def improvement_median(self) -> float:
        """Median improvement across all runs."""
        return statistics.median(self._improvements)

    @property
    def improvement_range(self) -> tuple[float, float]:
        """25th to 75th percentile range of improvements (more robust than min/max)."""
        return _quartiles_25_75(self._improvements)

    @property
    def current_ci_95(self) -> tuple[float, float]:
        """95% confidence interval for current implementation (ms)."""
        times_ms = self.current_times.iter().map(lambda t: t * 1e3).collect()
        return (times_ms.min(), times_ms.max())

    @property
    def new_pc_ci_95(self) -> tuple[float, float]:
        """95% confidence interval for new_pc implementation (ms)."""
        times_ms = self.new_pc_times.iter().map(lambda t: t * 1e3).collect()
        return (times_ms.min(), times_ms.max())


def _total_seconds(stmt: Callable[[], object], *, number: int) -> float:
    """Run statement `number` times and return total duration in seconds."""
    return timeit.timeit(stmt, number=number)


def _fmt_ms(seconds: float) -> str:
    return f"{seconds * 1e3:.3f} ms"


def _fmt_percent(percent: float) -> str:
    sign = "+" if percent >= 0 else ""
    color = "red" if percent >= 0 else "green"
    return f"[{color}]{sign}{percent:.2f}%[/{color}]"


def _print_table(cases: pc.Seq[Case]) -> None:
    """Print benchmark results using Rich table."""
    table = Table(
        title="Option & Result pipelines: current vs new_pc (median + Q1-Q3 range)"
    )
    table.add_column("Case", style="cyan")
    table.add_column("Current (ms)", justify="right", style="magenta")
    table.add_column("New PC (ms)", justify="right", style="magenta")
    table.add_column("Improvement %", justify="right", style="yellow")
    table.add_column("Q1-Q3 Range", justify="right", style="dim")
    cases.iter().sort(key=lambda x: x.name).iter().for_each(
        lambda c: table.add_row(
            c.name,
            _fmt_ms(c.current_median),
            _fmt_ms(c.new_pc_median),
            _fmt_percent(c.improvement_median),
            f"{c.improvement_range[0]:.1f}% to {c.improvement_range[1]:.1f}%",
        )
    )
    console = Console()
    console.print(table)


# Constants for benchmark conditions
DATA_SIZE = 100
NUM_ITERATIONS = 500
NUM_RUNS = 10

# Thresholds for various benchmarks
THRESHOLD_HIGH = 250
THRESHOLD_LOW = 100
THRESHOLD_MID = 50
THRESHOLD_NESTING = 450
MIN_NESTING = 5
THRESHOLD_RANGE = 400
THRESHOLD_MID_TRANSFORM = 250
THRESHOLD_MIN_SECONDARY = 25
THRESHOLD_LONG_FILTER = 500
THRESHOLD_LONG_MIN = 2
THRESHOLD_ZIP_A = 150
THRESHOLD_ZIP_B = 50
THRESHOLD_MIXED_ZIP = 50

# Initialize data once
_BENCH_DATA = list(range(DATA_SIZE))


def _pipeline_simple_map(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Simple pipeline: map an Option inside Iter."""

    def _process(x: int) -> int:
        def _to_option(v: int) -> OptionProto[Any]:
            return some_ctor(v * 2) if v % 2 == 0 else none_val

        return _to_option(x).map(lambda v: v + 10).unwrap_or(0)

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_filter_map(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Filter + map chain on Options."""

    def _process(x: int) -> int:
        def _to_option(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v > THRESHOLD_HIGH else none_val

        return (
            _to_option(x)
            .filter(lambda v: v % 3 == 0)
            .map(lambda v: v * 2)
            .unwrap_or(-1)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_and_then(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """and_then chains: flatten nested Options."""

    def _process(x: int) -> int:
        def _parse_even(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 2 == 0 else none_val

        def _div_by_3(v: int) -> OptionProto[Any]:
            return some_ctor(v // 3) if v % 3 == 0 else none_val

        return _parse_even(x).and_then(_div_by_3).unwrap_or(0)

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_zip(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Zip two Options in parallel."""

    def _process(x: int) -> int:
        def _left(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v < THRESHOLD_HIGH else none_val

        def _right(v: int) -> OptionProto[Any]:
            return some_ctor(v + 1000) if v > THRESHOLD_LOW else none_val

        left_opt = _left(x)
        right_opt = _right(x)
        return left_opt.zip(right_opt).map(lambda pair: pair[0] + pair[1]).unwrap_or(0)

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_complex(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Complex pipeline: multiple transformations, aggregations."""

    def _process(x: int) -> int:
        def _parse(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 2 == 0 else none_val

        def _validate(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v > THRESHOLD_MID else none_val

        return (
            _parse(x)
            .and_then(_validate)
            .map(lambda v: v * 3)
            .filter(lambda v: v > 0)
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_deep_nesting(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Deep nesting: multiple and_then chains."""

    def _process(x: int) -> int:
        def _step1(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 2 == 0 else none_val

        def _step2(v: int) -> OptionProto[Any]:
            return some_ctor(v + 1) if v < THRESHOLD_NESTING else none_val

        def _step3(v: int) -> OptionProto[Any]:
            return some_ctor(v * 2) if v % 3 == 0 else none_val

        def _step4(v: int) -> OptionProto[Any]:
            return some_ctor(v - 5) if v > MIN_NESTING else none_val

        return (
            _step1(x).and_then(_step2).and_then(_step3).and_then(_step4).unwrap_or(-1)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_long_chain(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Long chain: multiple map/filter operations."""

    def _process(x: int) -> int:
        def _to_option(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v > 0 else none_val

        return (
            _to_option(x)
            .map(lambda v: v + 1)
            .filter(lambda v: v % 2 == 0)
            .map(lambda v: v * 2)
            .filter(lambda v: v < THRESHOLD_LONG_FILTER)
            .map(lambda v: v // 2)
            .filter(lambda v: v > THRESHOLD_LONG_MIN)
            .map(lambda v: v * 3)
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_multi_validation(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Multi-validation: sequential validators with and_then."""

    def _process(x: int) -> int:
        def _is_even(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 2 == 0 else none_val

        def _in_range_low(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v >= THRESHOLD_LOW else none_val

        def _in_range_high(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v <= THRESHOLD_RANGE else none_val

        def _divisible_by_3(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 3 == 0 else none_val

        return (
            _is_even(x)
            .and_then(_in_range_low)
            .and_then(_in_range_high)
            .and_then(_divisible_by_3)
            .map(lambda v: v * 2)
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_mixed_ops(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Mixed operations: combination of map, filter, and_then, zip."""

    def _process(x: int) -> int:
        def _base(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v % 2 == 0 else none_val

        def _transform(v: int) -> OptionProto[Any]:
            return some_ctor(v * 2) if v < THRESHOLD_MID_TRANSFORM else none_val

        def _secondary(v: int) -> OptionProto[Any]:
            return some_ctor(v + 10) if v > THRESHOLD_MIN_SECONDARY else none_val

        base_opt = _base(x)
        secondary_opt = _secondary(x)
        return (
            base_opt.map(lambda v: v + 1)
            .and_then(_transform)
            .zip(secondary_opt)
            .map(lambda pair: pair[0] + pair[1])
            .filter(lambda v: v > THRESHOLD_MIXED_ZIP)
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _pipeline_chained_zips(
    some_ctor: Callable[[int], OptionProto[Any]], none_val: OptionProto[Any]
) -> int:
    """Chained zips: multiple zip operations in sequence."""

    def _process(x: int) -> int:
        def _opt_a(v: int) -> OptionProto[Any]:
            return some_ctor(v) if v < THRESHOLD_ZIP_A else none_val

        def _opt_b(v: int) -> OptionProto[Any]:
            return some_ctor(v + 5) if v > THRESHOLD_ZIP_B else none_val

        def _opt_c(v: int) -> OptionProto[Any]:
            return some_ctor(v * 2) if v % 2 == 0 else none_val

        a = _opt_a(x)
        b = _opt_b(x)
        c = _opt_c(x)
        return (
            a.zip(b)
            .map(lambda pair: pair[0] + pair[1])
            .and_then(lambda v: c.zip(some_ctor(v)))
            .map(lambda pair: pair[0] + pair[1])
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_simple_map(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Simple Result pipeline: map inside Iter."""

    def _process(x: int) -> int:
        def _to_result(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v * 2) if v % 2 == 0 else err_ctor("even_required")

        return _to_result(x).map(lambda v: v + 10).unwrap_or(0)

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_and_then(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Result and_then chains."""

    def _process(x: int) -> int:
        def _parse_even(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v) if v % 2 == 0 else err_ctor("not_even")

        def _div_by_3(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v // 3) if v % 3 == 0 else err_ctor("not_div_3")

        return _parse_even(x).and_then(_div_by_3).unwrap_or(0)

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_map_err(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Result with map_err error handling."""

    def _process(x: int) -> int:
        def _validate(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v) if v > THRESHOLD_HIGH else err_ctor("below_threshold")

        return (
            _validate(x)
            .map(lambda v: v * 2)
            .map_err(lambda _: 0)
            .or_else(lambda _: ok_ctor(0))
            .unwrap_or(-1)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_deep_chain(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Deep Result chains with multiple and_then operations."""

    def _process(x: int) -> int:
        def _step1(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v) if v % 2 == 0 else err_ctor("step1_fail")

        def _step2(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v + 1) if v < THRESHOLD_NESTING else err_ctor("step2_fail")

        def _step3(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v * 2) if v % 3 == 0 else err_ctor("step3_fail")

        def _step4(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v - 5) if v > MIN_NESTING else err_ctor("step4_fail")

        return (
            _step1(x).and_then(_step2).and_then(_step3).and_then(_step4).unwrap_or(-1)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_mixed(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Mixed Result operations with map, map_err, and or_else."""

    def _process(x: int) -> int:
        def _validate(v: int) -> ResultProto[Any, Any]:
            return ok_ctor(v) if v % 2 == 0 else err_ctor("validate_failed")

        def _transform(v: int) -> ResultProto[Any, Any]:
            return (
                ok_ctor(v * 2)
                if v < THRESHOLD_MID_TRANSFORM
                else err_ctor("transform_failed")
            )

        return (
            _validate(x)
            .map(lambda v: v + 1)
            .and_then(_transform)
            .map_err(lambda _: -10)
            .or_else(lambda _: ok_ctor(0))
            .map(lambda v: v * 3)
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_with_option_transpose(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Result pipeline with complex branching using and_then and map_or_else."""

    def _process(x: int) -> int:
        def _to_result_option(v: int) -> ResultProto[Any, Any]:
            if v % 2 == 0:
                return (
                    ok_ctor(v * 2) if v < THRESHOLD_NESTING else err_ctor("too_large")
                )
            return err_ctor("not_even")

        res = _to_result_option(x)
        # Chain with and_then for complex branching based on value
        return (
            res.and_then(lambda v: ok_ctor(v * 3 if v > THRESHOLD_LOW else v + 100))
            .or_else(lambda _: ok_ctor(-1))
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def _result_pipeline_is_ok_and_filtering(
    ok_ctor: Callable[[int], ResultProto[Any, Any]],
    err_ctor: Callable[[str], ResultProto[Any, Any]],
) -> int:
    """Result pipeline using complex predicate filtering with and_then."""

    def _process(x: int) -> int:
        def _validate_range(v: int) -> ResultProto[Any, Any]:
            return (
                ok_ctor(v)
                if THRESHOLD_LOW <= v <= THRESHOLD_RANGE
                else err_ctor(f"out_of_range:{v}")
            )

        # Chain with and_then for predicate check and conditional transform
        return (
            _validate_range(x)
            .and_then(
                lambda v: ok_ctor(v * 2)
                if v % 3 == 0
                else err_ctor("not_divisible_by_3")
            )
            .or_else(
                lambda e: ok_ctor(THRESHOLD_MID if "out_of_range" in str(e) else 0)
            )
            .unwrap_or(0)
        )

    return pc.Iter(_BENCH_DATA).map(_process).collect().sum()


def main() -> None:
    """Run realistic pipeline benchmarks with both Option and Result types."""

    def _run_benchmark_multiple(
        name: str,
        current_fn: Callable[[], Any],
        new_pc_fn: Callable[[], Any],
    ) -> Case:
        """Run a benchmark multiple times and collect all measurements."""
        current_times = pc.Vec[float].new()
        new_pc_times = pc.Vec[float].new()

        for _ in range(NUM_RUNS):
            current_times.append(_total_seconds(current_fn, number=NUM_ITERATIONS))
            new_pc_times.append(_total_seconds(new_pc_fn, number=NUM_ITERATIONS))

        return Case(
            name=name,
            current_times=current_times,
            new_pc_times=new_pc_times,
        )

    # Collect all benchmark cases: realistic complex pipelines only (Option + Result mix)
    cases = (
        pc.Iter(
            [
                # Complex Option pipelines
                (
                    "option/chained_zips",
                    lambda: _pipeline_chained_zips(pc.Some, pc.NONE),
                    lambda: _pipeline_chained_zips(new_pc.Some, new_pc.NONE),
                ),
                (
                    "option/complex",
                    lambda: _pipeline_complex(pc.Some, pc.NONE),
                    lambda: _pipeline_complex(new_pc.Some, new_pc.NONE),
                ),
                (
                    "option/deep_nesting",
                    lambda: _pipeline_deep_nesting(pc.Some, pc.NONE),
                    lambda: _pipeline_deep_nesting(new_pc.Some, new_pc.NONE),
                ),
                (
                    "option/long_chain",
                    lambda: _pipeline_long_chain(pc.Some, pc.NONE),
                    lambda: _pipeline_long_chain(new_pc.Some, new_pc.NONE),
                ),
                (
                    "option/mixed_ops",
                    lambda: _pipeline_mixed_ops(pc.Some, pc.NONE),
                    lambda: _pipeline_mixed_ops(new_pc.Some, new_pc.NONE),
                ),
                (
                    "option/multi_validation",
                    lambda: _pipeline_multi_validation(pc.Some, pc.NONE),
                    lambda: _pipeline_multi_validation(new_pc.Some, new_pc.NONE),
                ),
                # Complex Result pipelines
                (
                    "result/deep_chain",
                    lambda: _result_pipeline_deep_chain(pc.Ok, pc.Err),
                    lambda: _result_pipeline_deep_chain(new_pc.Ok, new_pc.Err),
                ),
                (
                    "result/mixed",
                    lambda: _result_pipeline_mixed(pc.Ok, pc.Err),
                    lambda: _result_pipeline_mixed(new_pc.Ok, new_pc.Err),
                ),
                (
                    "result/with_option_transpose",
                    lambda: _result_pipeline_with_option_transpose(pc.Ok, pc.Err),
                    lambda: _result_pipeline_with_option_transpose(
                        new_pc.Ok, new_pc.Err
                    ),
                ),
                (
                    "result/is_ok_and_filtering",
                    lambda: _result_pipeline_is_ok_and_filtering(pc.Ok, pc.Err),
                    lambda: _result_pipeline_is_ok_and_filtering(new_pc.Ok, new_pc.Err),
                ),
            ]
        )
        .map(lambda item: _run_benchmark_multiple(item[0], item[1], item[2]))
        .collect()
    )

    _print_table(cases)


if __name__ == "__main__":
    main()
