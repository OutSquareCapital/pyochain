"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import functools
import timeit
from collections.abc import Callable
from enum import IntEnum
from typing import Final, NamedTuple

import polars as pl
import typer
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

import pyochain as pc

type Row = tuple[str, str, str, int, float]
app = typer.Typer(help="Benchmarks for pyochain developments.")

CONSOLE: Final = Console()


class Runs(IntEnum):
    """Cost category for benchmarks, determining iteration counts."""

    FOCUSED = 20_000
    CHEAP = 5_000
    NORMAL = 2_500
    EXPENSIVE = 50


type BenchFn = Callable[[], object]


class Benchmark(NamedTuple):
    """A benchmark with both implementations."""

    category: str
    name: str
    n_runs: Runs
    old_fn: BenchFn
    new_fn: BenchFn


BENCHMARKS = pc.Vec[Benchmark].new()


def bench[T, R](
    category: str,
    *,
    old: T,
    new: T,
    n_runs: Runs = Runs.CHEAP,
) -> Callable[[Callable[[T], R]], Callable[[T], R]]:
    """Decorator to register a benchmark function for both old and new implementations."""

    def decorator(func: Callable[[T], R]) -> Callable[[T], R]:
        def old_fn() -> R:
            return func(old)

        def new_fn() -> R:
            return func(new)

        assert old_fn() == new_fn(), (
            f"{func.__name__}: Old and new implementations must produce the same result"
        )

        BENCHMARKS.append(Benchmark(category, func.__name__, n_runs, old_fn, new_fn))
        return func

    return decorator


def _collect_raw_timings() -> pc.Iter[Row]:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    n_benchmarks = BENCHMARKS.length()
    total_runs: int = BENCHMARKS.iter().map(lambda b: b.n_runs).sum() * 2
    CONSOLE.print(
        f"[dim]Found {n_benchmarks} benchmarks, {total_runs} total runs[/dim]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Running benchmarks...", total=total_runs)
        f = functools.partial(_run_benchmark, progress, task)

        return BENCHMARKS.iter().flat_map(
            lambda b: f(b, "new", b.new_fn).chain(f(b, "old", b.old_fn))
        )


def _run_benchmark(
    progress: Progress, task: TaskID, bench: Benchmark, impl_name: str, fn: BenchFn
) -> pc.Iter[Row]:
    def _update_progress(run_idx: int) -> tuple[str, str, str, int, float]:
        progress.update(
            task,
            description=f"[cyan]{bench.category}: {bench.name} ({impl_name})",
        )
        progress.advance(task)

        return (
            bench.category,
            bench.name,
            impl_name,
            run_idx,
            timeit.timeit(fn, number=10),
        )

    return pc.Iter(range(bench.n_runs)).map(_update_progress)


def _compute_all_stats(raw_rows: pc.Iter[Row]) -> pl.DataFrame:
    """Compute all stats from raw timings using Polars expressions + over."""
    time = pl.col("time")
    median = pl.col("median")
    group = ["category", "name"]
    return (
        pl.LazyFrame(
            raw_rows,
            schema=["category", "name", "impl", "run_idx", "time"],
            orient="row",
        )
        .group_by("category", "name", "impl")
        .agg(
            time.median().alias("median"),
            pl.len().alias("runs"),
        )
        .pipe(
            lambda stats: stats.filter(pl.col("impl").eq("new")).join(
                stats.filter(pl.col("impl").eq("old")).select(
                    *group, median.alias("old_median")
                ),
                on=group,
            )
        )
        .rename({"median": "new_median"})
        .drop("impl")
        .with_columns(
            pl.col("old_median").truediv(pl.col("new_median")).alias("ratio"),
        )
        .sort("ratio", descending=True)
        .collect()
    )


def _print_summary(pivoted: pl.DataFrame) -> None:
    """Print summary stats from pivoted DataFrame."""
    summary = pivoted.select(
        pl.col("ratio").median().alias("median_speedup"),
        pl.col("ratio").gt(1).sum().alias("wins"),
        pl.len().alias("total"),
    )
    median_speedup = summary.get_column("median_speedup").item(0)
    wins = summary.get_column("wins").item(0)
    total = summary.get_column("total").item(0)

    CONSOLE.print()
    CONSOLE.print(
        Text("Median speedup: ", style="bold").append(
            Text(f"{median_speedup:.2f}x", style="green bold")
        )
    )
    CONSOLE.print(
        Text("New wins: ", style="bold").append(Text(f"{wins}/{total}", style="cyan"))
    )


@app.command()
def all_benchmarks() -> None:
    """Run all benchmarks."""
    CONSOLE.print(Text("Running benchmarks...", style="bold blue"))
    CONSOLE.print()
    pivoted = _collect_raw_timings().into(_compute_all_stats)
    CONSOLE.print()
    table = _build_results_table(pivoted)
    CONSOLE.print(table)
    _print_summary(pivoted)
    CONSOLE.print()


def _build_results_table(pivoted: pl.DataFrame) -> Table:
    """Build Rich table directly from Polars columns."""
    table = Table(title="Benchmark Results")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Runs", justify="right", style="magenta")
    table.add_column("New (μs, median)", justify="right", style="green")
    table.add_column("Old (μs, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")
    pc.Iter(
        pivoted.select(
            "category",
            "name",
            pl.col("runs").cast(pl.String),
            pl.col("new_median").mul(1_000_000).round(2).cast(pl.String),
            pl.col("old_median").mul(1_000_000).round(2).cast(pl.String),
            pl.col("ratio").round(2).cast(pl.String).add("x").alias("ratio_str"),
            pl.when(pl.col("ratio").gt(1))
            .then(pl.lit("green bold"))
            .otherwise(pl.lit("red bold"))
            .alias("style"),
        ).iter_rows()
    ).for_each_star(
        lambda cat, name, runs, new_med, old_med, ratio_str, style: table.add_row(
            cat, name, runs, new_med, old_med, Text(ratio_str, style=style)
        )
    )

    return table


if __name__ == "__main__":
    app()
