"""Benchmarks for pyochain developments."""

import timeit
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, NamedTuple

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


@dataclass(slots=True)
class _BaseRow:
    category: str
    name: str


@dataclass(slots=True)
class Row(_BaseRow):
    """Raw row of timing data."""

    impl: str
    size: int
    run_idx: int
    time: float


@dataclass(slots=True)
class TableRow(_BaseRow):
    """Finalized row out from polars."""

    size: str
    runs: str
    new_med: str
    old_med: str
    pct_str: str
    style: str

    def add_to_table(self, table: Table) -> None:
        """Add this row to a Rich table."""
        return table.add_row(
            self.category,
            self.name,
            self.size,
            self.runs,
            self.new_med,
            self.old_med,
            Text(self.pct_str, style=self.style),
        )


type BenchFn = Callable[[], object]


class Variant(NamedTuple):
    """A specific benchmark variant size."""

    size: int
    n_runs: int
    old_fn: BenchFn
    new_fn: BenchFn


WARMUP_RUNS: Final = 5
CALLS_BY_RUN: Final = 10
TARGET_BENCH_SEC: Final = 1
MIN_RUNS: Final = 20


app = typer.Typer(help="Benchmarks for pyochain developments.")

CONSOLE: Final = Console()


class Benchmark(NamedTuple):
    """A benchmark with multiple data sizes."""

    category: str
    name: str
    variants: pc.Vec[Variant]


BENCHMARKS = pc.Vec[Benchmark].new()


def bench[P, T, R](
    category: str,
    *,
    old: Callable[[P], R],
    new: Callable[[P], R],
    data_gen: Callable[[pc.Iter[int]], P],
) -> Callable[..., Callable[[], None]]:
    """Decorator to register benchmarks with multiple data sizes."""

    def decorator(func: Callable[[], None]) -> Callable[[], None]:
        variants = pc.Vec[Variant].new()

        for size in (256, 512, 1024, 2048):
            data = pc.Iter(range(size)).into(data_gen)

            assert old(data) == new(data), (
                f"{func.__name__}: Old and new implementations must produce the same result for size {size}"
            )
            old_fn = partial(old, data)
            new_fn = partial(new, data)

            n_runs = max(_estimate_n_runs(old_fn), _estimate_n_runs(new_fn))

            variants.append(Variant(size, n_runs, old_fn, new_fn))

        BENCHMARKS.append(Benchmark(category, func.__name__, variants))
        return func

    return decorator


def _estimate_n_runs(fn: BenchFn) -> int:
    warmup_time = timeit.timeit(fn, number=WARMUP_RUNS) / WARMUP_RUNS
    est = int(TARGET_BENCH_SEC / 2 / warmup_time / CALLS_BY_RUN)
    return max(MIN_RUNS, est)


def _collect_raw_timings(benchmarks: pc.Vec[Benchmark]) -> pc.Seq[Row]:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    n_benchmarks = benchmarks.length()
    total_runs: int = (
        benchmarks.iter().flat_map(lambda b: b.variants).map(lambda v: v.n_runs).sum()
        * 2
    )
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
        f = partial(_run_variant, progress, task)
        return (
            benchmarks.iter()
            .flat_map(
                lambda bench: bench.variants.iter().flat_map(lambda v: f(v, bench))
            )
            .collect()
        )


def _run_variant(
    progress: Progress,
    task: Any,  # noqa: ANN401
    variant: Variant,
    bench: Benchmark,
) -> pc.Iter[Row]:
    def _update_progress(run_idx: int, impl: str, fn: BenchFn) -> Row:
        progress.update(
            task,
            description=f"[cyan]{bench.category}: {bench.name} @ {variant.size} ({impl})",
        )
        time_taken = timeit.timeit(fn, number=CALLS_BY_RUN)
        progress.advance(task)
        return Row(
            bench.category,
            bench.name,
            impl,
            variant.size,
            run_idx,
            time_taken,
        )

    def _run(runs: int, impl: str, fn: BenchFn) -> pc.Iter[Row]:
        return pc.Iter(range(runs)).map(
            lambda run_idx: _update_progress(run_idx, impl, fn)
        )

    return _run(variant.n_runs, "new", variant.new_fn).chain(
        _run(variant.n_runs, "old", variant.old_fn)
    )


def _compute_all_stats(raw_rows: pc.Seq[Row]) -> pl.LazyFrame:
    """Compute all stats from raw timings using Polars expressions + over."""
    group = ["category", "name", "size"]
    return (
        pl.LazyFrame(
            raw_rows,
            schema=["category", "name", "impl", "size", "run_idx", "time"],
            orient="row",
        )
        .group_by("category", "name", "size", "impl")
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
    table.add_column("Size", justify="right", style="magenta")
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
            pl.col("size").cast(pl.String),
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
        .map_star(TableRow)
        .for_each(lambda row: row.add_to_table(table))
    )


if __name__ == "__main__":
    app()
