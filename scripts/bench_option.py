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
