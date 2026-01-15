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

    FOCUSED = 10_000
    CHEAP = 5_000
    NORMAL = 2_500
    EXPENSIVE = 100


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


def _collect_raw_timings() -> pl.DataFrame:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    benchmark_pairs = pc.Dict[tuple[str, str], dict[Implementation, BenchFn]].new()
    raw_rows = pc.Vec[tuple[str, str, str, int, float]].new()

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
                    time_val = timeit.timeit(fn, number=meta.n_calls)
                    raw_rows.append(
                        (meta.category, meta.name, impl_name, run_idx, time_val)
                    )
                progress.advance(task)

    return pl.DataFrame(
        raw_rows.into(list),
        schema=["category", "name", "impl", "run_idx", "time"],
        orient="row",
    )


def _compute_all_stats(raw_df: pl.DataFrame) -> pl.DataFrame:
    """Compute all stats from raw timings using Polars expressions only."""
    time = pl.col("time")
    return (
        raw_df.group_by("category", "name", "impl")
        .agg(
            time.median().alias("median"),
            time.mean().alias("mean"),
            time.std().alias("std"),
            time.quantile(0.25).alias("q1"),
            time.quantile(0.75).alias("q3"),
        )
        .with_columns(
            pl.col("std").truediv(pl.col("median")).mul(100).alias("cv_pct"),
            pl.col("q3").sub(pl.col("q1")).truediv(pl.col("median")).alias("iqr_rel"),
        )
    )


def _pivot_for_comparison(stats_df: pl.DataFrame) -> pl.DataFrame:
    """Pivot stats to have old/new columns side by side with ratio."""
    return (
        stats_df.unpivot(
            index=["category", "name", "impl"],
            on=["median", "mean", "std", "q1", "q3", "cv_pct", "iqr_rel"],
        )
        .pivot(on="impl", index=["category", "name", "variable"], values="value")
        .pivot(on="variable", index=["category", "name"], values=["new", "old"])
        .with_columns(
            pl.col("old_median").truediv(pl.col("new_median")).alias("ratio"),
            pl.lit(1)
            .sub(pl.col("new_median").truediv(pl.col("old_median")))
            .mul(100)
            .alias("improvement_pct"),
        )
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


def _collect_focused_timings(
    old: BenchFn, new: BenchFn, runs: int, calls: int
) -> pl.LazyFrame:
    """Collect raw timings for focused benchmark."""
    raw_rows = pc.Vec[tuple[str, int, float]].new()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task("[cyan]Timing benchmarks...", total=runs * 2)

        for impl_name, fn in [("new", new), ("old", old)]:
            progress.update(task, description=f"[cyan]Timing {impl_name}")
            for run_idx in range(runs):
                time_val = timeit.timeit(fn, number=calls)
                raw_rows.append((impl_name, run_idx, time_val))
                progress.advance(task)

    return pl.LazyFrame(
        raw_rows.into(list),
        schema=["impl", "run_idx", "time"],
        orient="row",
    )


def _compute_focused_stats(raw_df: pl.LazyFrame) -> pl.DataFrame:
    """Compute stats for focused benchmark."""
    time = pl.col("time")
    return (
        raw_df.group_by("impl")
        .agg(
            time.median().alias("median"),
            time.mean().alias("mean"),
            time.std().alias("std"),
            time.quantile(0.25).alias("q1"),
            time.quantile(0.75).alias("q3"),
        )
        .with_columns(
            pl.col("std").truediv(pl.col("median")).mul(100).alias("cv_pct"),
            pl.col("q3").sub(pl.col("q1")).truediv(pl.col("median")).alias("iqr_rel"),
        )
        .unpivot(
            index="impl", on=["median", "mean", "std", "q1", "q3", "cv_pct", "iqr_rel"]
        )
        .collect()
        .pivot(on="impl", index="variable", values="value")
        .with_columns(
            pl.col("old").truediv(pl.col("new")).alias("ratio"),
        )
    )


def _build_focused_table(stats: pl.DataFrame) -> Table:
    """Build focused comparison table from pivoted stats."""
    table = Table()
    table.add_column("Metric", style="cyan")
    table.add_column("new", justify="right", style="green")
    table.add_column("old", justify="right", style="yellow")
    table.add_column("Relative", justify="right", style="magenta")

    def _get_stat(name: str) -> tuple[float, float, float]:
        return stats.filter(pl.col("variable").eq(name)).pipe(
            lambda row: (
                row.get_column("new").item(0),
                row.get_column("old").item(0),
                row.get_column("ratio").item(0),
            )
        )

    new_med, old_med, rel_median = _get_stat("median")
    new_mean, old_mean, rel_mean = _get_stat("mean")
    new_cv, old_cv, _ = _get_stat("cv_pct")
    new_iqr, old_iqr, _ = _get_stat("iqr_rel")

    new_faster = rel_median > 1
    winner = "new" if new_faster else "old"
    style = "green bold" if new_faster else "yellow bold"
    speedup = rel_median if new_faster else 1 / rel_median

    table.add_row(
        "Speedup",
        f"{1 / rel_median:.3f}x" if new_faster else "1.00x",
        "1.00x" if new_faster else f"{rel_median:.3f}x",
        Text(f"{winner} {speedup:.2f}x faster", style=style),
    )

    improvement_pct = (speedup - 1) * 100
    label = "faster" if new_faster else "slower"
    table.add_row(
        "Improvement",
        "—",
        "—",
        Text(f"{improvement_pct:+.1f}% ", style="dim").append(Text(label, style=style)),
    )

    table.add_row(
        "Median (rel)",
        "1.00",
        f"{rel_median:.3f}",
        f"{(1 - new_med / old_med) * 100:+.1f}%",
    )
    table.add_row(
        "Mean (rel)",
        "1.00",
        f"{rel_mean:.3f}",
        f"{(1 - new_mean / old_mean) * 100:+.1f}%",
    )
    table.add_row(
        "Variability (CV%)",
        f"{new_cv:.2f}%",
        f"{old_cv:.2f}%",
        f"{old_cv - new_cv:+.2f}%",
    )
    table.add_row(
        "IQR (rel)",
        f"{new_iqr:.4f}",
        f"{old_iqr:.4f}",
        f"{old_iqr - new_iqr:+.4f}",
    )
    return table


def _print_focused_conclusion(
    stats: pl.DataFrame, old_name: str, new_name: str
) -> None:
    """Print conclusion for focused benchmark."""
    rel_median = (
        stats.filter(pl.col("variable").eq("median")).get_column("ratio").item(0)
    )

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


def _run_focused_benchmark[T](old: partial[T], new: partial[T]) -> None:
    """Run focused, robust benchmark between two implementations."""
    runs = Runs.FOCUSED.value
    calls = runs // 10
    CONSOLE.print(Text("Running Focused Robustness Benchmark...", style="bold blue"))
    CONSOLE.print(
        Text(
            f"{runs:,} runs with {calls:,} calls in each for statistical significance...",
            style="dim",
        )
    )
    CONSOLE.print()
    stats = _collect_focused_timings(old, new, runs, calls).pipe(_compute_focused_stats)
    table = _build_focused_table(stats)
    CONSOLE.print(table)
    CONSOLE.print()
    _print_focused_conclusion(stats, old.func.__name__, new.func.__name__)


@app.command(name="all")
def all_benchmarks() -> None:
    """Run all benchmarks (default)."""
    CONSOLE.print(Text("Running Option benchmarks...", style="bold blue"))
    CONSOLE.print()
    pivoted = (
        _collect_raw_timings().pipe(_compute_all_stats).pipe(_pivot_for_comparison)
    )
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
    table.add_column("New (s, median)", justify="right", style="green")
    table.add_column("Old (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    (
        pc.Iter(pivoted.get_column("category"))
        .zip(pivoted.get_column("name"))
        .zip(pivoted.get_column("new_median"))
        .zip(pivoted.get_column("old_median"))
        .zip(pivoted.get_column("ratio"))
        .map(lambda t: (t[0][0][0][0], t[0][0][0][1], t[0][0][1], t[0][1], t[1]))
        .for_each(
            lambda row: table.add_row(
                str(row[0]),
                str(row[1]),
                f"{row[2]:.4f}",
                f"{row[3]:.4f}",
                Text(
                    f"{row[4]:.2f}x", style="green bold" if row[4] > 1 else "red bold"
                ),
            )
        )
    )
    return table


@app.command()
def focused() -> None:
    """Run focused build_args benchmark only."""
    data = pc.Seq(range(1_000_000))

    def _old() -> int:
        return data.iter().eq(range(1_000_000))

    def _new() -> int:
        return data.iter().eq_test(range(1_000_000))

    assert _old() == _new()

    _run_focused_benchmark(
        old=partial(_old),  # type: ignore[arg-type]
        new=partial(_new),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
