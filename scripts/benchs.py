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

type BenchFn = Callable[[], object]


class Runs(IntEnum):
    """Cost category for benchmarks, determining iteration counts."""

    FOCUSED = 20_000
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
    def from_times(cls, times: pc.Vec[float]) -> Self:
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

    def get_conclusion(self, old_name: str, new_name: str) -> Text:
        """Generate a conclusion text based on relative performance."""
        if self.rel_median > 1:
            return Text(
                f"✓ {new_name} is {self.rel_median:.2f}x faster ({self.improvement_pct:.1f}% improvement)",
                style="bold green",
            )
        return Text(
            f"✗ {old_name} is {1 / self.rel_median:.2f}x faster ({abs(self.improvement_pct):.1f}% regression)",
            style="bold yellow",
        )


class BenchmarkMetadata(NamedTuple):
    """Metadata for a benchmark function."""

    category: str
    name: str
    cost: Runs
    implementation: Implementation

    def get_median(self, fn: BenchFn) -> float:
        """Get the median execution time for the given function."""
        return (
            pc.Iter(range(self.cost.value))
            .map(lambda _: timeit.timeit(fn, number=self.n_calls))
            .into(statistics.median)
        )

    @property
    def n_calls(self) -> int:
        """Number of calls per timing iteration."""
        return self.cost.value // 10

    def to_result(self, rust_fn: BenchFn, python_fn: BenchFn) -> BenchmarkResult:
        """Run the benchmark and return the result."""
        rust_median = self.get_median(rust_fn)
        python_median = self.get_median(python_fn)

        return BenchmarkResult(
            category=self.category,
            name=self.name,
            rust_median=rust_median,
            python_median=python_median,
            speedup=python_median / rust_median,
        )


# timeit return total time, so we want to maximize number of runs to get stable median
TEST_VALUE: Final[int] = 42
CHAIN_VALUE: Final[int] = 5
CHAIN_THRESHOLD: Final[int] = 5


# Test data: large dataset with mixed None/values (realistic scenario)

NULLABLE_DATA: Final = (
    pc.Iter(range(100)).map(lambda x: x if x % 3 != 0 else None).collect()
)
INT_DATA_LARGE: Final = pc.Iter(range(100)).collect()


CONSOLE: Final = Console()
# Store all runs for each benchmark, then compute median
# Registry of benchmark functions with their metadata
BENCHMARK_REGISTRY = pc.Dict[BenchFn, BenchmarkMetadata].new()


# =============================================================================
# DECORATOR
# =============================================================================


def bench[O, N, R](
    category: str,
    name: str,
    *,
    old: O,
    new: N,
    cost: Runs = Runs.CHEAP,
) -> Callable[[Callable[[O | N], R]], Callable[[O | N], R]]:
    """Decorator to register a benchmark function for both old and new implementations."""

    def decorator(func: Callable[[O | N], R]) -> Callable[[O | N], R]:
        @wraps(func, updated=())
        def old_wrapper() -> R:
            return func(old)

        @wraps(func, updated=())
        def new_wrapper() -> R:
            return func(new)

        old_meta = BenchmarkMetadata(
            category=category,
            name=name,
            cost=cost,
            implementation=Implementation.PYTHON,
        )
        new_meta = BenchmarkMetadata(
            category=category, name=name, cost=cost, implementation=Implementation.RUST
        )

        BENCHMARK_REGISTRY[old_wrapper] = old_meta
        BENCHMARK_REGISTRY[new_wrapper] = new_meta

        return func

    return decorator


def _run_all_benchmarks() -> pc.Vec[BenchmarkResult]:
    """Run all registered benchmarks by pairing Rust and Python implementations."""
    benchmark_pairs: dict[tuple[str, str], dict[str, BenchFn]] = {}
    results = pc.Vec[BenchmarkResult].new()

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
            results.append(BENCHMARK_REGISTRY[rust_fn].to_result(rust_fn, python_fn))
            progress.advance(task)
    return results


def _display_results(results: pc.Vec[BenchmarkResult]) -> None:
    table = Table(title="Benchmark Results")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Rust (s, median)", justify="right", style="green")
    table.add_column("Python (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for result in results:
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
    medians = results.iter().map(lambda r: r.speedup).collect()
    median_speedup = medians.into(statistics.median)

    wins = medians.iter().filter(lambda x: x > 1).length()
    CONSOLE.print()
    summary_line = Text("Median speedup: ", style="bold") + Text(
        f"{median_speedup:.2f}x", style="green bold"
    )
    CONSOLE.print(summary_line)
    wins_line = Text("Rust wins: ", style="bold") + Text(
        f"{wins}/{results.length()}", style="cyan"
    )
    CONSOLE.print(wins_line)


def _run_focused_benchmark(old: partial[object], new: partial[object]) -> None:
    """Run focused, robust benchmark between two implementation."""
    old_name = old.func.__name__
    new_name = new.func.__name__
    calls = Runs.FOCUSED.value // 10
    CONSOLE.print(Text("Running Focused Robustness Benchmark...", style="bold blue"))
    CONSOLE.print(
        Text(
            f"{Runs.FOCUSED.value:,} runs with {calls:,} calls in each for statistical significance...",
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

        def _run_bench(name: str, fn: partial[object]) -> pc.Vec[float]:
            task = progress.add_task(f"[green]Benchmarking {name}...", total=calls)
            times = pc.Vec[float].new()
            for _ in range(calls):
                times.append(timeit.timeit(fn, number=Runs.FOCUSED))
                progress.advance(task)
            return times

    old_stats = _run_bench(new_name, new).into(Stats.from_times)
    new_stats = _run_bench(old_name, old).into(Stats.from_times)
    relative = RelativeStats.from_comparison(old_stats, new_stats)
    table = _get_table(relative, old_stats, new_stats)
    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.print(relative.get_conclusion(old_name, new_name))


def _get_table(relative: RelativeStats, old_stats: Stats, new_stats: Stats) -> Table:
    table = Table()
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
    return table


def main() -> None:
    """Run all benchmarks and display results."""
    CONSOLE.print(Text("Running Option benchmarks...", style="bold blue"))
    CONSOLE.print()
    _run_all_benchmarks().into(_display_results)


@app.command()
def all_benchmarks() -> None:
    """Run all benchmarks (default)."""
    main()


@app.command()
def focused() -> None:
    """Run focused build_args benchmark only."""
    from pyochain.rs import Pipeable

    def foo(x: object) -> object:
        return x

    a = Pipeable()
    _run_focused_benchmark(
        old=partial(a.into, foo),  # type: ignore[arg-type]
        new=partial(a.into_test, foo),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
