"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import timeit
from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from functools import partial, wraps
from typing import Final, NamedTuple

import polars as pl
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

app = typer.Typer(help="Benchmarks for pyochain developments.")

CONSOLE: Final = Console()


class Runs(IntEnum):
    """Cost category for benchmarks, determining iteration counts."""

    FOCUSED = 1000
    CHEAP = 5_000
    NORMAL = 2_500
    EXPENSIVE = 1000


class Implementation(StrEnum):
    """Implementation type for benchmarks."""

    OLD = auto()
    NEW = auto()


class BenchmarkMetadata(NamedTuple):
    """Metadata for a benchmark function."""

    category: str
    name: str
    cost: Runs
    implementation: Implementation

    @property
    def n_calls(self) -> int:
        """Number of calls per timing iteration."""
        return self.cost.value // 10


type BenchFn = Callable[[], object]
BENCHMARK_REGISTRY = pc.Dict[BenchFn, BenchmarkMetadata].new()


def bench[T, N, R](
    category: str,
    *,
    old: T,
    new: T,
    cost: Runs = Runs.CHEAP,
) -> Callable[[Callable[[T], R]], Callable[[T], R]]:
    """Decorator to register a benchmark function for both old and new implementations."""

    def decorator(func: Callable[[T], R]) -> Callable[[T], R]:
        @wraps(func, updated=())
        def old_wrapper() -> R:
            return func(old)

        @wraps(func, updated=())
        def new_wrapper() -> R:
            return func(new)

        old_meta = BenchmarkMetadata(
            category=category,
            name=func.__name__,
            cost=cost,
            implementation=Implementation.OLD,
        )
        new_meta = BenchmarkMetadata(
            category=category,
            name=func.__name__,
            cost=cost,
            implementation=Implementation.NEW,
        )
        BENCHMARK_REGISTRY.try_insert(old_wrapper, old_meta).expect(
            "Failed to register benchmark"
        )
        BENCHMARK_REGISTRY.try_insert(new_wrapper, new_meta).expect(
            "Failed to register benchmark"
        )

        return func

    return decorator


def _run_all_benchmarks() -> pl.DataFrame:
    """Run all registered benchmarks by pairing Rust and Python implementations."""
    benchmark_pairs = pc.Dict[tuple[str, str], dict[Implementation, BenchFn]].new()
    results = pc.Vec[dict[str, object]].new()

    for func, meta in BENCHMARK_REGISTRY.items().iter():
        key = (meta.category, meta.name)
        if key not in benchmark_pairs:
            benchmark_pairs[key] = {}
        benchmark_pairs[key][meta.implementation] = func

    benchmarks = (
        benchmark_pairs.values()
        .iter()
        .map(lambda impls: (impls[Implementation.NEW], impls[Implementation.OLD]))
        .collect()
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task(
            "[cyan]Running benchmarks...", total=len(benchmarks) * 2
        )
        for rust_fn, python_fn in benchmarks:
            meta = BENCHMARK_REGISTRY.get_item(rust_fn).unwrap()
            n_calls = meta.n_calls

            progress.update(
                task, description=f"[cyan]{meta.category}: {meta.name} (New)"
            )
            rust_times = _run_timing_measurements(rust_fn, meta.cost.value, n_calls)
            progress.advance(task)

            progress.update(
                task, description=f"[cyan]{meta.category}: {meta.name} (Old)"
            )
            python_times = _run_timing_measurements(python_fn, meta.cost.value, n_calls)
            progress.advance(task)

            rust_median = pl.Series(rust_times).median()
            python_median = pl.Series(python_times).median()

            results.append(
                {
                    "category": meta.category,
                    "name": meta.name,
                    "rust_median": rust_median,
                    "python_median": python_median,
                }
            )

    return pl.DataFrame(results).with_columns(
        pl.col("python_median").truediv(pl.col("rust_median")).alias("ratio")
    )


def _display_results(results: pl.DataFrame) -> None:
    """Display benchmark results table and summary."""
    table = Table(title="Benchmark Results")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Rust (s, median)", justify="right", style="green")
    table.add_column("Python (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    def _add_row(row: dict[str, object]) -> None:
        ratio: float = row["ratio"]  # type: ignore[assignment]
        style = "green bold" if ratio > 1 else "red bold"
        table.add_row(
            str(row["category"]),
            str(row["name"]),
            f"{row['rust_median']:.4f}",
            f"{row['python_median']:.4f}",
            Text(f"{ratio:.2f}x", style=style),
        )

    pc.Iter(results.iter_rows(named=True)).for_each(_add_row)
    CONSOLE.print(table)

    summary = results.select(
        pl.col("ratio").median().alias("median_speedup"),
        pl.col("ratio").gt(1).sum().alias("wins"),
        pl.len().alias("total"),
    ).row(0, named=True)

    CONSOLE.print()
    summary_line = Text("Median speedup: ", style="bold") + Text(
        f"{summary['median_speedup']:.2f}x", style="green bold"
    )
    CONSOLE.print(summary_line)
    wins_line = Text("Rust wins: ", style="bold") + Text(
        f"{summary['wins']}/{summary['total']}", style="cyan"
    )
    CONSOLE.print(wins_line)


def _run_timing_measurements(fn: BenchFn, runs: int, iterations: int) -> pc.Seq[float]:
    """Run timing measurements for a function and return execution times."""
    return (
        pc.Iter(range(runs))
        .map(lambda _: timeit.timeit(fn, number=iterations))
        .collect()
    )


def _run_timing_measurements_with_progress(  # noqa: PLR0913
    fn: BenchFn,
    runs: int,
    iterations: Runs,
    progress: Progress,
    task_id: object,
    description: str,
) -> pc.Seq[float]:
    """Run timing measurements with progress bar updates."""
    progress.update(task_id, description=description)  # type: ignore[arg-type]

    def _measure(_: int) -> float:
        result = timeit.timeit(fn, number=iterations.value // 10)
        progress.advance(task_id)  # type: ignore[arg-type]
        return result

    return pc.Iter(range(runs)).map(_measure).collect()


def _run_focused_benchmark[T](old: partial[T], new: partial[T]) -> None:
    """Run focused, robust benchmark between two implementations."""
    calls = Runs.FOCUSED.value // 10
    CONSOLE.print(Text("Running Focused Robustness Benchmark...", style="bold blue"))
    CONSOLE.print(
        Text(
            f"{Runs.FOCUSED.value:,} runs with {calls:,} calls in each for statistical significance...",
            style="dim",
        )
    )
    _display_speed_comparison(old.func.__name__, new.func.__name__, old, new, calls)


def _display_speed_comparison(
    old_name: str, new_name: str, old: BenchFn, new: BenchFn, calls: int
) -> None:
    """Display speed comparison between two implementations using Polars for stats."""
    CONSOLE.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Timing benchmarks...", total=calls * 2)
        old_times = _run_timing_measurements_with_progress(
            old, calls, Runs.FOCUSED, progress, task, f"[cyan]Timing {old_name}"
        )
        new_times = _run_timing_measurements_with_progress(
            new, calls, Runs.FOCUSED, progress, task, f"[cyan]Timing {new_name}"
        )

    comparison_df = _compute_comparison_stats(old_times, new_times)
    table = _build_comparison_table(comparison_df)
    CONSOLE.print(table)
    CONSOLE.print()
    _print_conclusion(comparison_df, old_name, new_name)


def _compute_comparison_stats(
    old_times: pc.Seq[float], new_times: pc.Seq[float]
) -> pl.DataFrame:
    """Compute comparison statistics using Polars long format."""
    time_col = pl.col("time")
    sources = (
        pc.Iter.once("old")
        .cycle()
        .take(old_times.length())
        .chain(pc.Iter.once("new").cycle().take(new_times.length()))
        .collect()
    )
    times = old_times.iter().chain(new_times).collect()
    return (
        pl.DataFrame({"source": sources, "time": times})
        .group_by("source")
        .agg(
            time_col.median().alias("median"),
            time_col.mean().alias("mean"),
            time_col.std().alias("std"),
            time_col.quantile(0.25).alias("q1"),
            time_col.quantile(0.75).alias("q3"),
        )
        .with_columns(
            pl.col("std").truediv(pl.col("median")).mul(100).alias("cv_pct"),
            pl.col("q3").sub(pl.col("q1")).truediv(pl.col("median")).alias("iqr_rel"),
        )
    )


def _build_comparison_table(stats_df: pl.DataFrame) -> Table:
    """Build Rich table from pre-computed stats DataFrame."""
    stats = (
        pc.Iter(stats_df.iter_rows(named=True))
        .map(lambda r: (r["source"], r))
        .collect(pc.Dict)
    )
    old = stats.get_item("old").unwrap()
    new = stats.get_item("new").unwrap()
    rel_median = old["median"] / new["median"]
    rel_mean = old["mean"] / new["mean"]
    improvement_pct = (1 - new["median"] / old["median"]) * 100

    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("new", justify="right", style="green")
    table.add_column("old", justify="right", style="yellow")
    table.add_column("Relative", justify="right", style="magenta")

    new_faster = rel_median > 1
    winner, loser_style = (
        ("new", "green bold") if new_faster else ("old", "yellow bold")
    )
    speedup = rel_median if new_faster else 1 / rel_median
    table.add_row(
        "Speedup",
        f"{1 / rel_median:.3f}x" if new_faster else "1.00x",
        "1.00x" if new_faster else f"{rel_median:.3f}x",
        Text(f"{winner} {speedup:.2f}x faster", style=loser_style),
    )

    improvement_value = (speedup - 1) * 100
    label = "faster" if new_faster else "slower"
    table.add_row(
        "Improvement",
        "—",
        "—",
        Text(f"{improvement_value:+.1f}% ", style="dim")
        + Text(label, style=loser_style),
    )

    table.add_row(
        "Median (rel)", "1.00", f"{rel_median:.3f}", f"{improvement_pct:+.1f}%"
    )
    table.add_row(
        "Mean (rel)",
        "1.00",
        f"{rel_mean:.3f}",
        f"{(1 - new['mean'] / old['mean']) * 100:+.1f}%",
    )
    table.add_row(
        "Variability (CV%)",
        f"{new['cv_pct']:.2f}%",
        f"{old['cv_pct']:.2f}%",
        f"{old['cv_pct'] - new['cv_pct']:+.2f}%",
    )
    table.add_row(
        "IQR (rel)",
        f"{new['iqr_rel']:.4f}",
        f"{old['iqr_rel']:.4f}",
        f"{old['iqr_rel'] - new['iqr_rel']:+.4f}",
    )
    return table


def _print_conclusion(stats_df: pl.DataFrame, old_name: str, new_name: str) -> None:
    """Print conclusion message based on comparison stats."""
    stats = pc.Iter(stats_df.iter_rows(named=True)).into(
        lambda rows: pc.Dict({r["source"]: r["median"] for r in rows})
    )
    rel_median = stats.get_item("old").unwrap() / stats.get_item("new").unwrap()

    if rel_median > 1:
        CONSOLE.print(
            Text(
                f"✓ {new_name} is +{(rel_median - 1) * 100:.1f}% faster",
                style="bold green",
            )
        )
    else:
        CONSOLE.print(
            Text(
                f"✗ {old_name} is +{(1 - rel_median) * 100:.1f}% faster",
                style="bold red",
            )
        )


@app.command(name="all")
def all_benchmarks() -> None:
    """Run all benchmarks (default)."""
    CONSOLE.print(Text("Running Option benchmarks...", style="bold blue"))
    CONSOLE.print()
    timing_results = _run_all_benchmarks()
    CONSOLE.print()
    _display_results(timing_results)
    CONSOLE.print()


@app.command()
def focused() -> None:
    """Run focused build_args benchmark only."""

    def _old() -> int:
        return 1

    def _new() -> int:
        return 1

    assert _old() == _new()

    _run_focused_benchmark(
        old=partial(_old),  # type: ignore[arg-type]
        new=partial(_new),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
