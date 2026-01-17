"""Benchmarks for pyochain developments."""

import functools
import timeit
from collections.abc import Callable
from dataclasses import dataclass
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

WARMUP_RUNS: Final = 5
CALLS_BY_RUN: Final = 10
TARGET_BENCH_SEC: Final = 1.0
MIN_RUNS: Final = 20


app = typer.Typer(help="Benchmarks for pyochain developments.")

CONSOLE: Final = Console()


type BenchFn = Callable[[], object]


class Benchmark(NamedTuple):
    """A benchmark with both implementations."""

    category: str
    name: str
    n_runs: int
    old_fn: BenchFn
    new_fn: BenchFn


BENCHMARKS = pc.Vec[Benchmark].new()


def bench[T, R](
    category: str,
    *,
    old: T,
    new: T,
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
        n_runs = max(_estimate_n_runs(old_fn), _estimate_n_runs(new_fn))

        BENCHMARKS.append(Benchmark(category, func.__name__, n_runs, old_fn, new_fn))
        return func

    return decorator


def _estimate_n_runs(fn: Callable[[], object]) -> int:
    warmup_time = timeit.timeit(fn, number=WARMUP_RUNS) / WARMUP_RUNS
    est = int(TARGET_BENCH_SEC / 2 / warmup_time / CALLS_BY_RUN)
    return max(MIN_RUNS, est)


def _collect_raw_timings(benchmarks: pc.Vec[Benchmark]) -> pc.Seq[Row]:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    n_benchmarks = benchmarks.length()
    total_runs: int = benchmarks.iter().map(lambda b: b.n_runs).sum() * 2
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

        return (
            BENCHMARKS.iter()
            .flat_map(lambda b: f(b, "new", b.new_fn).chain(f(b, "old", b.old_fn)))
            .collect()
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
            timeit.timeit(fn, number=CALLS_BY_RUN),
        )

    return pc.Iter(range(bench.n_runs)).map(_update_progress)


def _compute_all_stats(raw_rows: pc.Seq[Row]) -> pl.LazyFrame:
    """Compute all stats from raw timings using Polars expressions + over."""
    group = ["category", "name"]
    return (
        pl.LazyFrame(
            raw_rows,
            schema=["category", "name", "impl", "run_idx", "time"],
            orient="row",
        )
        .group_by("category", "name", "impl")
        .agg(
            pl.col("time").median().alias("median"),
            pl.len().alias("runs"),
        )
        .pipe(
            lambda stats: stats.filter(pl.col("impl").eq("new")).join(
                stats.filter(pl.col("impl").eq("old")).select(
                    *group, pl.col("median").alias("old_median")
                ),
                on=group,
            )
        )
        .rename({"median": "new_median"})
        .drop("impl")
        .with_columns(
            pl.col("old_median")
            .truediv("new_median")
            .sub(1)
            .mul(100)
            .alias("pct_change"),
        )
        .sort("pct_change", descending=True)
    )


def _try_collect(lf: pl.LazyFrame) -> pc.Result[pl.DataFrame, str]:
    """Try to collect a LazyFrame, with error handling."""
    try:
        return pc.Ok(lf.collect())
    except (
        pl.exceptions.ColumnNotFoundError,
        pl.exceptions.InvalidOperationError,
    ) as e:
        return pc.Err(f"{e}")


@dataclass(slots=True)
class SummaryStats:
    """Summary statistics for benchmark results."""

    median_speedup: float
    wins: int
    total: int

    def _speedup(self) -> Text:
        return Text("Median speedup: ", style="bold").append(
            f"{self.median_speedup:+.1f}%",
            style="green bold" if self.median_speedup >= 0 else "red bold",
        )

    def _win_rate(self) -> Text:
        return Text("New wins: ", style="bold").append(
            f"{self.wins}/{self.total}", style="cyan"
        )

    def show(self) -> None:
        """Display the summary statistics to the console."""
        CONSOLE.print()
        CONSOLE.print(self._speedup())
        CONSOLE.print(self._win_rate())


def _print_summary(pivoted: pl.DataFrame) -> None:
    """Print summary stats from pivoted DataFrame."""
    return (
        pivoted.select(
            pl.col("pct_change").median().alias("median_speedup"),
            pl.col("pct_change").gt(0).sum().alias("wins"),
            pl.len().alias("total"),
        )
        .pipe(
            lambda df: SummaryStats(
                median_speedup=df.get_column("median_speedup").item(0),
                wins=df.get_column("wins").item(0),
                total=df.get_column("total").item(0),
            )
        )
        .show()
    )


@app.command()
def main() -> None:
    """Run benchmarks."""
    CONSOLE.print("Running benchmarks...", style="bold blue")
    CONSOLE.print()
    df = _run_pipeline()
    CONSOLE.print()
    table = _build_results_table()
    df.pipe(_fill_table, table)
    CONSOLE.print(table)
    df.pipe(_print_summary)


def _run_pipeline() -> pl.DataFrame:
    return (
        BENCHMARKS.ok_or("No benchmarks registered!")
        .map(_collect_raw_timings)
        .map(_compute_all_stats)
        .and_then(_try_collect)
        .unwrap()
    )


def _build_results_table() -> Table:
    """Build Rich table directly from Polars columns."""
    table = Table(title="Benchmark Results")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Runs", justify="right", style="magenta")
    table.add_column("New (μs, median)", justify="right", style="green")
    table.add_column("Old (μs, median)", justify="right", style="yellow")
    table.add_column("Change", justify="right")

    return table


def _fill_table(df: pl.DataFrame, table: Table) -> None:
    def _format_median(expr: pl.Expr) -> pl.Expr:
        return expr.mul(1_000_000).round(2).cast(pl.String)

    return (
        df.select(
            "category",
            "name",
            pl.col("runs").cast(pl.String),
            pl.col("new_median").pipe(_format_median),
            pl.col("old_median").pipe(_format_median),
            pl.col("pct_change").round(1).cast(pl.String).add("%").alias("pct_str"),
            pl.when(pl.col("pct_change").gt(0))
            .then(pl.lit("green bold"))
            .otherwise(pl.lit("red bold"))
            .alias("style"),
        )
        .pipe(lambda df: pc.Iter(df.iter_rows()))
        .for_each_star(
            lambda cat, name, runs, new_med, old_med, pct_str, style: table.add_row(
                cat, name, runs, new_med, old_med, Text(pct_str, style=style)
            )
        )
    )


if __name__ == "__main__":
    app()
