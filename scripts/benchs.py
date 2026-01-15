"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import timeit
from collections.abc import Callable
from enum import IntEnum, StrEnum, auto
from functools import wraps
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

    FOCUSED = 10_000
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


type BenchFn = Callable[[], object]
type BenchRegistery = pc.Dict[Implementation, BenchFn]
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


def _collect_raw_timings() -> pl.LazyFrame:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    benchmark_pairs = _get_pairs()
    raw_rows = pc.Vec[tuple[str, str, str, int, float]].new()

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
            "[cyan]Running benchmarks...", total=benchmarks.length() * 2
        )

        for new_fn, old_fn in benchmarks:
            meta = BENCHMARK_REGISTRY.get_item(new_fn).unwrap()

            for impl_name, fn in [("new", new_fn), ("old", old_fn)]:
                progress.update(
                    task,
                    description=f"[cyan]{meta.category}: {meta.name} ({impl_name})",
                )
                for run_idx in range(meta.cost.value):
                    time_val = timeit.timeit(fn, number=10)
                    raw_rows.append(
                        (meta.category, meta.name, impl_name, run_idx, time_val)
                    )
                progress.advance(task)

    return pl.LazyFrame(
        raw_rows.into(list),
        schema=["category", "name", "impl", "run_idx", "time"],
        orient="row",
    )


def _get_pairs() -> pc.Dict[tuple[str, str], pc.Dict[Implementation, BenchFn]]:
    benchmark_pairs = pc.Dict[tuple[str, str], pc.Dict[Implementation, BenchFn]].new()
    (
        BENCHMARK_REGISTRY.items()
        .ok_or("Error, no benchmarks registered")
        .unwrap()
        .iter()
        .for_each_star(
            lambda func, meta: benchmark_pairs.try_insert(
                (meta.category, meta.name), pc.Dict[Implementation, BenchFn].new()
            )
            .ok()
            .map(lambda d: d.insert(meta.implementation, func))
        )
    )
    return benchmark_pairs


def _compute_all_stats(raw_df: pl.LazyFrame) -> pl.DataFrame:
    """Compute all stats from raw timings using Polars expressions + over."""
    time = pl.col("time")
    median = pl.col("median")
    group = ["category", "name"]
    return (
        raw_df.group_by("category", "name", "impl")
        .agg(
            time.median().alias("median"),
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


@app.command(name="all")
def all_benchmarks() -> None:
    """Run all benchmarks (default)."""
    CONSOLE.print(Text("Running Option benchmarks...", style="bold blue"))
    CONSOLE.print()
    pivoted = _collect_raw_timings().pipe(_compute_all_stats)
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
    table.add_column("New (ms, median)", justify="right", style="green")
    table.add_column("Old (ms, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")
    pc.Iter(
        pivoted.select(
            "category",
            "name",
            pl.col("new_median").mul(1000).round(2).cast(pl.String),
            pl.col("old_median").mul(1000).round(2).cast(pl.String),
            pl.col("ratio").round(2).cast(pl.String).add("x").alias("ratio_str"),
            pl.when(pl.col("ratio").gt(1))
            .then(pl.lit("green bold"))
            .otherwise(pl.lit("red bold"))
            .alias("style"),
        ).iter_rows()
    ).for_each_star(
        lambda cat, name, new_med, old_med, ratio_str, style: table.add_row(
            cat, name, new_med, old_med, Text(ratio_str, style=style)
        )
    )

    return table


if __name__ == "__main__":
    app()
