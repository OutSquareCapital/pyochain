import timeit
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, NamedTuple, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

import pyochain as pc

type BenchFn = Callable[[], object]


WARMUP_RUNS: Final = 5
CALLS_BY_RUN: Final = 10
TARGET_BENCH_SEC: Final = 1
MIN_RUNS: Final = 20

CONSOLE: Final = Console()


class Variant(NamedTuple):
    """A specific benchmark variant size."""

    size: int
    n_runs: int
    fn: BenchFn

    @classmethod
    def from_fn(cls, fn: BenchFn, size: int) -> Self:
        """Estimate number of runs needed for benchmark variant."""
        warmup_time = timeit.timeit(fn, number=WARMUP_RUNS) / WARMUP_RUNS
        est = int(TARGET_BENCH_SEC / 2 / warmup_time / CALLS_BY_RUN)
        return cls(size, max(MIN_RUNS, est), fn)


class Benchmark(NamedTuple):
    """A benchmark with multiple data sizes."""

    category: str
    name: str
    variants: pc.Vec[Variant]


@dataclass(slots=True)
class Row:
    """Raw row of timing data."""

    category: str
    name: str
    size: int
    run_idx: int
    time: float


BENCHMARKS = pc.Vec[Benchmark].new()


def bench[P](
    *, gen: Callable[[pc.Iter[int]], P] = lambda size: size.collect()
) -> Callable[[Callable[[P], object]], Callable[[P], object]]:
    """Decorator to register benchmarks with multiple data sizes."""

    def decorator(func: Callable[[P], object]) -> Callable[[P], object]:
        variants = pc.Vec[Variant].new()
        for size in (256, 512, 1024, 2048):
            data = pc.Iter(range(size)).into(gen)
            variants.append(Variant.from_fn(partial(func, data), size))

        BENCHMARKS.append(
            Benchmark(func.__qualname__.split(".")[0], func.__name__, variants)
        )
        return func

    return decorator


def collect_raw_timings(benchmarks: pc.Vec[Benchmark]) -> pc.Seq[Row]:
    """Collect raw timing data for all benchmarks. Stats computed at the end."""
    total_runs: int = (
        benchmarks.iter().flat_map(lambda b: b.variants).map(lambda v: v.n_runs).sum()
    )
    CONSOLE.print(
        f"Found {benchmarks.length()} benchmarks, {total_runs} total runs",
        style="bold white",
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
    def _update_progress(run_idx: int, fn: BenchFn) -> Row:
        progress.update(
            task,
            description=f"[cyan]{bench.category}: {bench.name} @ {variant.size}",
        )
        time_taken = timeit.timeit(fn, number=CALLS_BY_RUN)
        progress.advance(task)
        return Row(
            bench.category,
            bench.name,
            variant.size,
            run_idx,
            time_taken,
        )

    return pc.Iter(range(variant.n_runs)).map(
        lambda run_idx: _update_progress(run_idx, variant.fn)
    )
