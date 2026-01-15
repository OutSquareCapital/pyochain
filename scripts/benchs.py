"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import statistics
import timeit
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum, StrEnum, auto
from functools import partial, wraps
from typing import Annotated, Final, NamedTuple, Self

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

type BenchFn = Callable[[], object]


class Runs(IntEnum):
    """Cost category for benchmarks, determining iteration counts."""

    FOCUSED = 1000
    CHEAP = 5_000
    NORMAL = 2_500
    EXPENSIVE = 1000


class Implementation(StrEnum):
    """Implementation type for benchmarks."""

    RUST = auto()
    PYTHON = auto()


@dataclass(slots=True)
class BenchResult:
    """Base class for benchmark results."""

    category: str
    name: str
    rust_result: float
    python_result: float

    @property
    def style(self) -> str:
        """Get style based on speedup."""
        return "green bold" if self.ratio > 1 else "red bold"

    def to_row(self) -> tuple[str, str, str, str, Text]:
        """Convert the result to a table row."""
        return (
            self.category,
            self.name,
            f"{self.rust_result:.4f}",
            f"{self.python_result:.4f}",
            Text(f"{self.ratio:.2f}x", style=self.style),
        )

    @property
    def ratio(self) -> float:
        """Get the ratio of Python to Rust results."""
        return self.python_result / self.rust_result


class MemoryStats(NamedTuple):
    """Statistical summary of memory measurements."""

    median: float
    mean: float
    min: float
    max: float

    @classmethod
    def from_memories(cls, memories: pc.Vec[float]) -> Self:
        """Compute stats from a list of memory measurements (in MB)."""
        return cls(
            statistics.median(memories),
            statistics.mean(memories),
            pc.Seq(memories).iter().min(),
            pc.Seq(memories).iter().max(),
        )


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
            improvement_pct = (self.rel_median - 1) * 100
            return Text(
                f"✓ {new_name} is +{improvement_pct:.1f}% faster",
                style="bold green",
            )
        improvement_pct = (1 - self.rel_median) * 100
        return Text(
            f"✗ {old_name} is +{improvement_pct:.1f}% faster",
            style="bold red",
        )


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


CONSOLE: Final = Console()
BENCHMARK_REGISTRY = pc.Dict[BenchFn, BenchmarkMetadata].new()


def bench[O, N, R](
    category: str,
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
            name=func.__name__,
            cost=cost,
            implementation=Implementation.PYTHON,
        )
        new_meta = BenchmarkMetadata(
            category=category,
            name=func.__name__,
            cost=cost,
            implementation=Implementation.RUST,
        )

        BENCHMARK_REGISTRY[old_wrapper] = old_meta
        BENCHMARK_REGISTRY[new_wrapper] = new_meta

        return func

    return decorator


def _run_all_benchmarks() -> pc.Vec[BenchResult]:
    """Run all registered benchmarks by pairing Rust and Python implementations."""
    benchmark_pairs: dict[tuple[str, str], dict[str, BenchFn]] = {}
    timing_results = pc.Vec[BenchResult].new()
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
        task = progress.add_task(
            "[cyan]Running benchmarks...", total=len(benchmarks) * 2
        )
        for rust_fn, python_fn in benchmarks:
            meta = BENCHMARK_REGISTRY[rust_fn]
            n_calls = meta.n_calls

            # Timing benchmark
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

            timing_results.append(
                BenchResult(
                    category=meta.category,
                    name=meta.name,
                    rust_result=rust_times.into(statistics.median),
                    python_result=python_times.into(statistics.median),
                )
            )

    return timing_results


def _display_results(results: pc.Vec[BenchResult]) -> None:
    table = Table(title="Benchmark Results")
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Rust (s, median)", justify="right", style="green")
    table.add_column("Python (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for result in results:
        table.add_row(*result.to_row())

    CONSOLE.print(table)

    # Summary
    medians = results.iter().map(lambda r: r.ratio).collect()
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


def _run_timing_measurements(fn: BenchFn, runs: int, iterations: int) -> pc.Vec[float]:
    """Run timing measurements for a function and return execution times."""
    times = pc.Vec[float].new()
    for _ in range(runs):
        times.append(timeit.timeit(fn, number=iterations))
    return times


def _run_timing_measurements_with_progress(
    fn: BenchFn,
    runs: int,
    iterations: Runs,
    progress: Progress,
    task_id: object,
    description: str,
) -> pc.Vec[float]:
    """Run timing measurements with progress bar updates."""
    times = pc.Vec[float].new()
    progress.update(task_id, description=description)  # type: ignore[arg-type]
    for _ in range(runs):
        times.append(timeit.timeit(fn, number=iterations.value // 10))
        progress.advance(task_id)  # type: ignore[arg-type]
    return times


def _run_memory_measurements(fn: BenchFn, runs: int, iterations: int) -> pc.Vec[float]:
    """Run memory measurements for a function and return peak memory values."""
    memories = pc.Vec[float].new()
    tracemalloc.start()

    for _ in range(runs):
        # Get baseline before test
        baseline_current, _ = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()

        # Run test iterations
        for _ in range(iterations):
            fn()

        # Measure peak relative to baseline
        _current, peak = tracemalloc.get_traced_memory()
        net_peak = max(0.0, peak - baseline_current)
        memories.append(net_peak / 1024 / 1024)

    tracemalloc.stop()
    return memories


def _run_focused_benchmark(
    old: partial[object], new: partial[object], *, memory: bool
) -> None:
    """Run focused, robust benchmark between two implementations with timing and memory."""
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
    _display_speed_comparison(old_name, new_name, old, new, calls)
    if memory:
        _display_memory_comparison(old_name, new_name, old, new, calls)


def _display_speed_comparison(
    old_name: str, new_name: str, old: BenchFn, new: BenchFn, calls: int
) -> None:
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
    old_stats = old_times.into(Stats.from_times)
    new_stats = new_times.into(Stats.from_times)
    relative = RelativeStats.from_comparison(old_stats, new_stats)
    table = _get_table(relative, old_stats, new_stats)
    CONSOLE.print(table)
    CONSOLE.print()
    CONSOLE.print(relative.get_conclusion(old_name, new_name))


def _display_memory_comparison(
    old_name: str, new_name: str, old: BenchFn, new: BenchFn, calls: int
) -> None:
    # Memory measurements
    CONSOLE.print()
    CONSOLE.print(Text("Memory Profiling...", style="bold blue"))
    old_mem_stats = _run_memory_measurements(old, calls, Runs.FOCUSED).into(
        MemoryStats.from_memories
    )
    new_mem_stats = _run_memory_measurements(new, calls, Runs.FOCUSED).into(
        MemoryStats.from_memories
    )
    CONSOLE.print()
    mem_table = Table()
    mem_table.add_column("Metric", style="cyan")
    mem_table.add_column(old_name, justify="right", style="yellow")
    mem_table.add_column(new_name, justify="right", style="green")
    mem_table.add_column("Ratio", justify="right", style="magenta")

    ratio = (
        old_mem_stats.median / new_mem_stats.median if new_mem_stats.median > 0 else 1.0
    )
    ratio_style = "green bold" if ratio > 1 else "red bold"

    mem_table.add_row(
        "Median (MB)",
        f"{old_mem_stats.median:.4f}",
        f"{new_mem_stats.median:.4f}",
        Text(f"{ratio:.4f}x", style=ratio_style),
    )
    mem_table.add_row(
        "Mean (MB)",
        f"{old_mem_stats.mean:.4f}",
        f"{new_mem_stats.mean:.4f}",
        f"{old_mem_stats.mean / new_mem_stats.mean if new_mem_stats.mean > 0 else 1.0:.4f}x",
    )
    mem_table.add_row(
        "Min (MB)",
        f"{old_mem_stats.min:.4f}",
        f"{new_mem_stats.min:.4f}",
        f"{old_mem_stats.min / new_mem_stats.min if new_mem_stats.min > 0 else 1.0:.4f}x",
    )
    mem_table.add_row(
        "Max (MB)",
        f"{old_mem_stats.max:.4f}",
        f"{new_mem_stats.max:.4f}",
        f"{old_mem_stats.max / new_mem_stats.max if new_mem_stats.max > 0 else 1.0:.4f}x",
    )
    CONSOLE.print(mem_table)


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
        new_speedup = f"{1 / relative.rel_median:.3f}x"
        old_speedup = "1.00x"
    else:
        speedup_msg = Text("old ", style="yellow bold") + Text(
            f"{1 / relative.rel_median:.2f}x faster", style="yellow bold"
        )
        new_speedup = "1.00x"
        old_speedup = f"{relative.rel_median:.3f}x"

    table.add_row("Speedup", new_speedup, old_speedup, speedup_msg)

    if relative.rel_median > 1:
        improvement_label = Text("faster", style="green bold")
        improvement_value = (relative.rel_median - 1) * 100
    else:
        improvement_label = Text("slower", style="yellow bold")
        improvement_value = (1 - relative.rel_median) * 100
    improvement_text = (
        Text(f"{improvement_value:+.1f}% ", style="dim") + improvement_label
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
def focused(
    *,
    mem: Annotated[bool, typer.Option(help="Run memory benchmarks")] = False,
) -> None:
    """Run focused build_args benchmark only."""

    def _old():
        return 1

    def _new():
        return 1

    assert _old() == _new()

    _run_focused_benchmark(
        old=partial(_old),  # type: ignore[arg-type]
        new=partial(_new),  # type: ignore[arg-type]
        memory=mem,
    )


if __name__ == "__main__":
    app()
