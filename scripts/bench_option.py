"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import statistics
import timeit
from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from functools import partial, wraps
from typing import Final, NamedTuple, Self

import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

import pyochain as pc
from pyochain import old_option, old_result

app = typer.Typer(help="Option type benchmarks: Rust vs Python")


class Runs(IntEnum):
    """Cost category for benchmarks, determining iteration counts."""

    CHEAP = 5_000
    NORMAL = 2_500
    EXPENSIVE = 500


class Implementation(StrEnum):
    """Implementation type for benchmarks."""

    RUST = auto()
    PYTHON = auto()


class BenchmarkResult(NamedTuple):
    """Result of a single benchmark comparison."""

    category: str
    name: str
    rust_median: float
    python_median: float
    speedup: float


class Stats(NamedTuple):
    """Statistical summary of benchmark results."""

    median: float
    mean: float
    stddev: float
    q1: float
    q3: float

    @classmethod
    def from_times(cls, times: list[float]) -> Self:
        """Compute stats from a list of times."""
        return cls(
            statistics.median(times),
            statistics.mean(times),
            statistics.stdev(times),
            sorted(times)[len(times) // 4],
            sorted(times)[3 * len(times) // 4],
        )


class RelativeStats(NamedTuple):
    """Relative statistical summary compared to a baseline."""

    rel_median: float
    rel_mean: float
    improvement_pct: float
    rel_stddev_new: float
    rel_stddev_old: float
    q1_rel: float
    q3_rel: float

    @classmethod
    def from_comparison(cls, old_stats: Stats, new_stats: Stats) -> Self:
        """Compute relative stats between old and new benchmark results."""
        return cls(
            (old_stats.median / new_stats.median),
            (old_stats.mean / new_stats.mean),
            ((old_stats.median - new_stats.median) / old_stats.median) * 100,
            (new_stats.stddev / new_stats.median) * 100,
            (old_stats.stddev / old_stats.median) * 100,
            old_stats.q1 / new_stats.q1,
            old_stats.q3 / new_stats.q3,
        )


class BenchmarkMetadata(NamedTuple):
    """Metadata for a benchmark function."""

    category: str
    name: str
    cost: Runs
    implementation: Implementation


# timeit return total time, so we want to maximize number of runs to get stable median
TEST_VALUE: Final[int] = 42
CHAIN_VALUE: Final[int] = 5
CHAIN_THRESHOLD: Final[int] = 5


# Test data: large dataset with mixed None/values (realistic scenario)
NULLABLE_DATA: Final = [x if x % 3 != 0 else None for x in range(100)]
INT_DATA_LARGE: Final = list(range(100))
type BenchFn = Callable[[], object]

CONSOLE: Final = Console()
# Store all runs for each benchmark, then compute median
RESULTS: list[BenchmarkResult] = []
# Registry of benchmark functions with their metadata
BENCHMARK_REGISTRY: dict[BenchFn, BenchmarkMetadata] = {}


# =============================================================================
# DECORATOR
# =============================================================================


def bench(
    category: str,
    name: str,
    implementation: Implementation,
    cost: Runs = Runs.CHEAP,
) -> Callable[[BenchFn], BenchFn]:
    """Decorator to register a benchmark function with its metadata.

    Args:
        category (str): The category of the benchmark (e.g., "Instantiation").
        name (str): The name of the benchmark (e.g., "Some(value)").
        implementation (str): The implementation type ("rust" or "python").
        cost (Runs): The cost category, defaults to CHEAP.

    Returns:
        Callable: The decorated function.

    Examples:
    ```python
    @bench("Instantiation", "Some(value)", "rust")
    def bench_rust_some_direct() -> object:
        return pc.Some(TEST_VALUE)
    ```
    """

    def decorator(func: BenchFn) -> BenchFn:
        metadata = BenchmarkMetadata(
            category=category, name=name, cost=cost, implementation=implementation
        )
        BENCHMARK_REGISTRY[func] = metadata

        @wraps(func)
        def wrapper() -> object:
            return func()

        return wrapper

    return decorator


# =============================================================================
# 1. INSTANTIATION
# =============================================================================


@bench("Instantiation", "Some(value)", Implementation.RUST, Runs.CHEAP)
def bench_rust_some_direct() -> object:
    """Direct Some instantiation via Rust."""
    return pc.Some(TEST_VALUE)


@bench("Instantiation", "Some(value)", Implementation.PYTHON, Runs.CHEAP)
def bench_python_some_direct() -> object:
    """Direct Some instantiation via Python."""
    return old_option.Some(TEST_VALUE)


@bench("Instantiation", "Dispatch to Some", Implementation.RUST, Runs.CHEAP)
def bench_rust_option_dispatch_some() -> object:
    """Option.__new__ dispatching to Some (Rust)."""
    return pc.Option(TEST_VALUE)


@bench("Instantiation", "Dispatch to Some", Implementation.PYTHON, Runs.CHEAP)
def bench_python_option_dispatch_some() -> object:
    """Option.__new__ dispatching to Some (Python)."""
    return old_option.Option(TEST_VALUE)


@bench("Instantiation", "Dispatch to None", Implementation.RUST, Runs.CHEAP)
def bench_rust_option_dispatch_none() -> pc.Option[object]:
    """Option.__new__ dispatching to None (Rust)."""
    return pc.Option(None)


@bench("Instantiation", "Dispatch to None", Implementation.PYTHON, Runs.CHEAP)
def bench_python_option_dispatch_none() -> old_option.Option[object]:
    """Option.__new__ dispatching to None (Python)."""
    return old_option.Option(None)


RUST_SOME: Final = pc.Some(TEST_VALUE)
PYTHON_SOME: Final = old_option.Some(TEST_VALUE)
RUST_NONE: Final = pc.NONE
PYTHON_NONE: Final = old_option.NONE
RUST_SOME_OTHER: Final = pc.Some(TEST_VALUE)
PYTHON_SOME_OTHER: Final = old_option.Some(TEST_VALUE)
RUST_SOME_DIFF: Final = pc.Some(99)
PYTHON_SOME_DIFF: Final = old_option.Some(99)


# =============================================================================
# 3. EQUALITY CHECKS
# =============================================================================


@bench("Equality Checks", "__eq__", Implementation.RUST, Runs.CHEAP)
def bench_rust_eq_same() -> bool:
    """Rust == check (same value)."""
    return RUST_SOME == RUST_SOME_OTHER


@bench("Equality Checks", "__eq__", Implementation.PYTHON, Runs.CHEAP)
def bench_python_eq_same() -> bool:
    """Python == check (same value)."""
    return PYTHON_SOME == PYTHON_SOME_OTHER


@bench("Equality Checks", "eq_method", Implementation.RUST, Runs.CHEAP)
def bench_rust_eq_method_same() -> bool:
    """Rust .eq() method (same value) - avoids isinstance."""
    return RUST_SOME.eq(RUST_SOME_OTHER)


@bench("Equality Checks", "eq_method", Implementation.PYTHON, Runs.CHEAP)
def bench_python_eq_method_same() -> bool:
    """Python .eq() method (same value) - uses is_some() instead of isinstance."""
    return PYTHON_SOME.eq(PYTHON_SOME_OTHER)


# =============================================================================
# 4. MAP WITH CLOSURES
# =============================================================================


def _identity[T](x: T) -> T:
    return x


def _add_ten(x: int) -> int:
    return x + 10


@bench("Map with Closures", "map (identity)", Implementation.RUST, Runs.CHEAP)
def bench_rust_map_identity() -> object:
    """Rust map with identity closure."""
    return RUST_SOME.map(_identity)


@bench("Map with Closures", "map (identity)", Implementation.PYTHON, Runs.CHEAP)
def bench_python_map_identity() -> object:
    """Python map with identity closure."""
    return PYTHON_SOME.map(_identity)


@bench("Map with Closures", "map simple add", Implementation.RUST, Runs.CHEAP)
def bench_rust_map_function() -> object:
    """Rust map with named function."""
    return RUST_SOME.map(_add_ten)


@bench("Map with Closures", "map simple add", Implementation.PYTHON, Runs.CHEAP)
def bench_python_map_function() -> object:
    """Python map with named function."""
    return PYTHON_SOME.map(_add_ten)


def _wrap_add_ten(x: int) -> pc.Some[int]:
    return pc.Some(x + 10)


def _wrap_add_ten_py(x: int) -> old_option.Some[int]:
    return old_option.Some(x + 10)


# =============================================================================
# 8. CHAINED OPERATIONS
# =============================================================================
def _repr(opt: object, suffix: str) -> str:
    return repr(opt) + suffix


@bench(
    "Chained Operations", "map -> filter -> map", Implementation.RUST, Runs.EXPENSIVE
)
def bench_rust_chain() -> object:
    """Rust chained operations."""
    return (
        pc.Some(CHAIN_VALUE)
        .map(lambda x: x * 2)
        .filter(lambda x: x > CHAIN_THRESHOLD)
        .map(lambda x: x + 1)
        .and_then(_wrap_add_ten)
        .into(_repr, "!")
    )


@bench(
    "Chained Operations", "map -> filter -> map", Implementation.PYTHON, Runs.EXPENSIVE
)
def bench_python_chain() -> object:
    """Python chained operations."""
    return (
        old_option.Some(CHAIN_VALUE)
        .map(lambda x: x * 2)
        .filter(lambda x: x > CHAIN_THRESHOLD)
        .map(lambda x: x + 1)
        .and_then(_wrap_add_ten_py)
        .into(_repr, "!")
    )


# =============================================================================
# 9. ITER WITH OPTIONS
# =============================================================================


@bench("Iter with Options", "Iter.map(Option)", Implementation.RUST, Runs.EXPENSIVE)
def bench_rust_iter_map_option() -> object:
    """Rust: Iter.map converting nullable to Option."""
    return pc.Iter(NULLABLE_DATA).map(pc.Option).collect()


@bench("Iter with Options", "Iter.map(Option)", Implementation.PYTHON, Runs.EXPENSIVE)
def bench_python_iter_map_option() -> object:
    """Python: Iter.map converting nullable to Option."""
    return pc.Iter(NULLABLE_DATA).map(old_option.Option).collect()


@bench(
    "Iter with Options", "Iter.filter_map (simple)", Implementation.RUST, Runs.EXPENSIVE
)
def bench_rust_iter_filter_map_simple() -> object:
    """Rust: Iter.filter_map with simple transformation."""
    return (
        pc.Iter(NULLABLE_DATA)
        .filter_map(lambda x: pc.Option(x).map(lambda v: v * 2).map(str))
        .collect()
    )


@bench(
    "Iter with Options",
    "Iter.filter_map (simple)",
    Implementation.PYTHON,
    Runs.EXPENSIVE,
)
def bench_python_iter_filter_map_simple() -> object:
    """Python: Iter.filter_map with simple transformation."""
    return (
        pc.Iter(NULLABLE_DATA)
        .filter_map(lambda x: old_option.Option(x).map(lambda v: v * 2).map(str))  # type: ignore[arg-type]
        .collect()  # type: ignore[return-value]
    )


@bench(
    "Iter with Options",
    "Iter.map -> filter_map -> map",
    Implementation.RUST,
    Runs.EXPENSIVE,
)
def bench_rust_iter_chain_with_filter_map() -> object:
    """Rust: Complex chain with map -> filter_map -> map."""
    return (
        pc.Iter(NULLABLE_DATA)
        .map(pc.Option)
        .filter_map(
            lambda opt: opt.map(lambda x: x * 2)
            .filter(lambda x: x > 100)
            .ok_or("Error")
            .ok()
        )
        .collect()
    )


@bench(
    "Iter with Options",
    "Iter.map -> filter_map -> map",
    Implementation.PYTHON,
    Runs.EXPENSIVE,
)
def bench_python_iter_chain_with_filter_map() -> object:
    """Python: Complex chain with map -> filter_map -> map."""
    return (
        pc.Iter(NULLABLE_DATA)
        .map(old_option.Option)
        .filter_map(
            lambda opt: opt.map(lambda x: x * 2)
            .filter(lambda x: x > 100)
            .ok_or("Error")
            .ok()  # type: ignore[return-value]
        )
        .collect()
    )


# =============================================================================
# 10. COMPLEX METHODS
# =============================================================================

RUST_NESTED_SOME: Final[pc.Option[pc.Option[int]]] = pc.Some(pc.Some(TEST_VALUE))
PYTHON_NESTED_SOME: Final[old_option.Option[old_option.Option[int]]] = old_option.Some(
    old_option.Some(TEST_VALUE)
)
RUST_SOME_TUPLE: Final[pc.Option[tuple[int, int]]] = pc.Some((10, 20))
PYTHON_SOME_TUPLE: Final[old_option.Option[tuple[int, int]]] = old_option.Some((10, 20))
RUST_SOME_OK: Final[pc.Option[pc.Result[int, str]]] = pc.Some(pc.Ok(TEST_VALUE))
PYTHON_SOME_OK: Final[old_option.Option[old_result.Result[int, str]]] = old_option.Some(
    old_result.Ok(TEST_VALUE)
)
RUST_OK_SOME: Final[pc.Result[pc.Option[int], object]] = pc.Ok(pc.Some(TEST_VALUE))
PYTHON_OK_SOME: Final[old_result.Result[old_option.Option[int], object]] = (
    old_result.Ok(old_option.Some(TEST_VALUE))
)


@bench("Complex Methods", "flatten", Implementation.RUST, Runs.CHEAP)
def bench_rust_flatten() -> object:
    """Rust flatten nested Option."""
    return RUST_NESTED_SOME.flatten()


@bench("Complex Methods", "flatten", Implementation.PYTHON, Runs.CHEAP)
def bench_python_flatten() -> object:
    """Python flatten nested Option."""
    return PYTHON_NESTED_SOME.flatten()


@bench("Complex Methods", "unzip", Implementation.RUST, Runs.CHEAP)
def bench_rust_unzip() -> object:
    """Rust unzip Option of tuple."""
    return RUST_SOME_TUPLE.unzip()


@bench("Complex Methods", "unzip", Implementation.PYTHON, Runs.CHEAP)
def bench_python_unzip() -> object:
    """Python unzip Option of tuple."""
    return PYTHON_SOME_TUPLE.unzip()


@bench("Complex Methods", "zip", Implementation.RUST, Runs.CHEAP)
def bench_rust_zip() -> object:
    """Rust zip two Options."""
    return RUST_SOME.zip(RUST_SOME_OTHER)


@bench("Complex Methods", "zip", Implementation.PYTHON, Runs.CHEAP)
def bench_python_zip() -> object:
    """Python zip two Options."""
    return PYTHON_SOME.zip(PYTHON_SOME_OTHER)


def _sum_two(x: int, y: int) -> int:
    return x + y


@bench("Complex Methods", "zip_with", Implementation.RUST, Runs.CHEAP)
def bench_rust_zip_with() -> object:
    """Rust zip_with two Options."""
    return RUST_SOME.zip_with(RUST_SOME_OTHER, _sum_two)


@bench("Complex Methods", "zip_with", Implementation.PYTHON, Runs.CHEAP)
def bench_python_zip_with() -> object:
    """Python zip_with two Options."""
    return PYTHON_SOME.zip_with(PYTHON_SOME_OTHER, _sum_two)


@bench("Complex Methods", "transpose (Option->Result)", Implementation.RUST, Runs.CHEAP)
def bench_rust_transpose_option_result() -> object:
    """Rust transpose Option of Result to Result of Option."""
    return RUST_SOME_OK.transpose()


@bench(
    "Complex Methods", "transpose (Option->Result)", Implementation.PYTHON, Runs.CHEAP
)
def bench_python_transpose_option_result() -> object:
    """Python transpose Option of Result to Result of Option."""
    return PYTHON_SOME_OK.transpose()


@bench("Complex Methods", "transpose (Result->Option)", Implementation.RUST, Runs.CHEAP)
def bench_rust_transpose_result_option() -> object:
    """Rust transpose Result of Option to Option of Result."""
    return RUST_OK_SOME.transpose()


@bench(
    "Complex Methods", "transpose (Result->Option)", Implementation.PYTHON, Runs.CHEAP
)
def bench_python_transpose_result_option() -> object:
    """Python transpose Result of Option to Option of Result."""
    return PYTHON_OK_SOME.transpose()


# =============================================================================
# MAIN
# =============================================================================


def bench_one(
    rust_fn: BenchFn,
    python_fn: BenchFn,
) -> None:
    """Run a single benchmark multiple times and store median results.

    Uses metadata from the BENCHMARK_REGISTRY to determine category, name, and iteration counts.

    Args:
        rust_fn (Callable): The Rust implementation benchmark function.
        python_fn (Callable): The Python implementation benchmark function.

    """
    rust_meta = BENCHMARK_REGISTRY[rust_fn]

    # Use the cost from rust_meta (should be same for both)
    n_calls = rust_meta.cost.value // 10

    rust_times = [
        timeit.timeit(rust_fn, number=n_calls) for _ in range(rust_meta.cost.value)
    ]
    python_times = [
        timeit.timeit(python_fn, number=n_calls) for _ in range(rust_meta.cost.value)
    ]
    rust_median = statistics.median(rust_times)
    python_median = statistics.median(python_times)
    speedup = python_median / rust_median
    RESULTS.append(
        BenchmarkResult(
            category=rust_meta.category,
            name=rust_meta.name,
            rust_median=rust_median,
            python_median=python_median,
            speedup=speedup,
        )
    )


def _run_all_benchmarks() -> None:
    """Run all registered benchmarks by pairing Rust and Python implementations."""
    # Group functions by (category, name) to pair Rust and Python
    benchmark_pairs: dict[tuple[str, str], dict[str, BenchFn]] = {}

    for func, meta in BENCHMARK_REGISTRY.items():
        key = (meta.category, meta.name)
        if key not in benchmark_pairs:
            benchmark_pairs[key] = {}
        benchmark_pairs[key][meta.implementation] = func

    # Validate and create benchmark list
    benchmarks: list[tuple[BenchFn, BenchFn]] = []
    for (category, name), impls in benchmark_pairs.items():
        if "rust" not in impls or "python" not in impls:
            CONSOLE.print(
                f"[yellow]Warning: Skipping {category}/{name} - missing implementation[/yellow]"
            )
            continue
        benchmarks.append((impls["rust"], impls["python"]))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=len(benchmarks))
        for rust_fn, python_fn in benchmarks:
            meta = BENCHMARK_REGISTRY[rust_fn]
            progress.update(task, description=f"[cyan]{meta.category}: {meta.name}")
            bench_one(rust_fn, python_fn)
            progress.advance(task)


def _display_results() -> None:
    """Display benchmark results in a formatted table."""
    table = Table(title="Option Type Benchmark Results (Rust vs Python)")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Rust (s, median)", justify="right", style="green")
    table.add_column("Python (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for result in RESULTS:
        speedup_style = "green bold" if result.speedup > 1 else "red bold"
        speedup_str = Text(f"{result.speedup:.2f}x", style=speedup_style)
        table.add_row(
            result.category,
            result.name,
            f"{result.rust_median:.4f}",
            f"{result.python_median:.4f}",
            speedup_str,
        )

    CONSOLE.print(table)

    # Summary
    median_speedup = statistics.median([r.speedup for r in RESULTS])
    wins = sum(1 for r in RESULTS if r.speedup > 1)
    CONSOLE.print()
    summary_line = Text("Median speedup: ", style="bold") + Text(
        f"{median_speedup:.2f}x", style="green bold"
    )
    CONSOLE.print(summary_line)
    wins_line = Text("Rust wins: ", style="bold") + Text(
        f"{wins}/{len(RESULTS)}", style="cyan"
    )
    CONSOLE.print(wins_line)


def _run_focused_benchmark(old: partial[object], new: partial[object]) -> None:
    """Run focused, robust benchmark between two implementation."""
    old_name = old.func.__name__
    new_name = new.func.__name__
    CONSOLE.print(Text("Running Focused Robustness Benchmark...", style="bold blue"))
    CONSOLE.print(
        Text(
            f"{Runs.CHEAP.value:,} runs with {Runs.CHEAP.value // 10:,} calls in each for statistical significance...",
            style="dim",
        )
    )
    CONSOLE.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        calls = Runs.CHEAP.value // 10
        # Benchmark new
        task_new = progress.add_task(f"[green]Benchmarking {new_name}...", total=calls)
        new_times: list[float] = []
        for _ in range(calls):
            new_times.append(timeit.timeit(new, number=Runs.CHEAP))
            progress.advance(task_new)

        # Benchmark old
        task_old = progress.add_task(f"[yellow]Benchmarking {old_name}...", total=calls)
        old_times: list[float] = []
        for _ in range(calls):
            old_times.append(timeit.timeit(old, number=Runs.CHEAP))
            progress.advance(task_old)

    # Benchmark eq_direct (FFI direct access, no type checking)
    old_stats = Stats.from_times(old_times)
    new_stats = Stats.from_times(new_times)
    relative = RelativeStats.from_comparison(old_stats, new_stats)

    table = Table(
        title=f"{new_name} vs {old_name}\n{Runs.CHEAP.value:,} ops x {calls} repeats"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("new", justify="right", style="green")
    table.add_column("old", justify="right", style="yellow")
    table.add_column("Relative", justify="right", style="magenta")

    if relative.rel_median > 1:
        speedup_msg = Text("new ", style="green bold") + Text(
            f"{relative.rel_median:.2f}x faster", style="green bold"
        )
    else:
        speedup_msg = Text("old ", style="yellow bold") + Text(
            f"{1 / relative.rel_median:.2f}x faster", style="yellow bold"
        )

    table.add_row("Speedup", "1.00x", f"{1 / relative.rel_median:.3f}x", speedup_msg)

    if relative.improvement_pct > 0:
        improvement_label = Text("faster", style="green bold")
    else:
        improvement_label = Text("slower", style="yellow bold")
    improvement_text = (
        Text(f"{relative.improvement_pct:+.1f}% ", style="dim") + improvement_label
    )
    table.add_row("Improvement", "—", "—", improvement_text)

    # Median
    table.add_row(
        "Median (rel)",
        "1.00",
        f"{relative.rel_median:.3f}",
        f"{relative.improvement_pct:+.1f}%",
    )

    # Mean
    table.add_row(
        "Mean (rel)",
        "1.00",
        f"{old_stats.mean / new_stats.mean:.3f}",
        f"{((old_stats.mean - new_stats.mean) / old_stats.mean * 100):+.1f}%",
    )

    # Variability (CV%)
    table.add_row(
        "Variability (CV%)",
        f"{relative.rel_stddev_new:.2f}%",
        f"{relative.rel_stddev_old:.2f}%",
        f"{relative.rel_stddev_old - relative.rel_stddev_new:+.2f}%",
    )

    # IQR
    table.add_row(
        "IQR (rel)",
        f"{(new_stats.q3 - new_stats.q1) / new_stats.median:.4f}",
        f"{(old_stats.q3 - old_stats.q1) / old_stats.median:.4f}",
        f"{((old_stats.q3 - old_stats.q1) / old_stats.median - (new_stats.q3 - new_stats.q1) / new_stats.median):+.4f}",
    )

    CONSOLE.print(table)
    CONSOLE.print()
    if relative.rel_median > 1:
        CONSOLE.print(
            Text(
                f"✓ {new_name} is {relative.rel_median:.2f}x faster ({relative.improvement_pct:.1f}% improvement)",
                style="bold green",
            )
        )
    else:
        CONSOLE.print(
            Text(
                f"✗ {old_name} is {1 / relative.rel_median:.2f}x faster ({abs(relative.improvement_pct):.1f}% regression)",
                style="bold yellow",
            )
        )


def main() -> None:
    """Run all benchmarks and display results."""
    CONSOLE.print(Text("Running Option benchmarks...", style="bold blue"))
    CONSOLE.print()
    _run_all_benchmarks()
    _display_results()


@app.command()
def all_benchmarks() -> None:
    """Run all benchmarks (default)."""
    main()


@app.command()
def focused() -> None:
    """Run focused build_args benchmark only."""
    # Test: map_star - call1 vs call(..., None)
    CONSOLE.print(
        "\n[bold cyan]Test: map_star - call1 (optimized) vs call(..., None)[/bold cyan]"
    )
    rust_some_tuple = pc.Some((10, 20))
    _run_focused_benchmark(
        old=partial(rust_some_tuple.map_star_old, lambda x, y: x + y),  # type: ignore[arg-type]
        new=partial(rust_some_tuple.map_star, lambda x, y: x + y),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
