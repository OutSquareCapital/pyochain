"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import statistics
import timeit
from collections.abc import Callable
from functools import partial
from typing import Final

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

import pyochain as pc
from pyochain import old_option

app = typer.Typer(help="Option type benchmarks: Rust vs Python")

# timeit return total time, so we want to maximize number of runs to get stable median
N_RUNS: Final[int] = 1_000  # nb of times to re-run the work for median val
N_REPEATS: Final[int] = 1000  # nb of times the function is called in timeit
TEST_VALUE: Final[int] = 42
CHAIN_VALUE: Final[int] = 5
CHAIN_THRESHOLD: Final[int] = 5


CONSOLE: Final = Console()
# Store all runs for each benchmark, then compute median
RESULTS: list[tuple[str, str, float, float, float]] = []


# =============================================================================
# 1. INSTANTIATION
# =============================================================================


def bench_rust_some_direct() -> object:
    """Direct Some instantiation via Rust."""
    return pc.Some(TEST_VALUE)


def bench_python_some_direct() -> object:
    """Direct Some instantiation via Python."""
    return old_option.Some(TEST_VALUE)


def bench_rust_option_dispatch_some() -> object:
    """Option.__new__ dispatching to Some (Rust)."""
    return pc.Option(TEST_VALUE)


def bench_python_option_dispatch_some() -> object:
    """Option.__new__ dispatching to Some (Python)."""
    return old_option.Option(TEST_VALUE)


def bench_rust_option_dispatch_none() -> pc.Option[object]:
    """Option.__new__ dispatching to None (Rust)."""
    return pc.Option(None)


def bench_python_option_dispatch_none() -> old_option.Option[object]:
    """Option.__new__ dispatching to None (Python)."""
    return old_option.Option(None)


# =============================================================================
# 2. TYPE CHECKING (is_some / is_none)
# =============================================================================

RUST_SOME: Final = pc.Some(TEST_VALUE)
PYTHON_SOME: Final = old_option.Some(TEST_VALUE)
RUST_NONE: Final = pc.NONE
PYTHON_NONE: Final = old_option.NONE


def bench_rust_is_some() -> bool:
    """Rust is_some() check."""
    return RUST_SOME.is_some()


def bench_python_is_some() -> bool:
    """Python is_some() check."""
    return PYTHON_SOME.is_some()


def bench_rust_is_none() -> bool:
    """Rust is_none() check."""
    return RUST_NONE.is_none()


def bench_python_is_none() -> bool:
    """Python is_none() check."""
    return PYTHON_NONE.is_none()


# =============================================================================
# 3. EQUALITY CHECKS
# =============================================================================

RUST_SOME_OTHER: Final = pc.Some(TEST_VALUE)
PYTHON_SOME_OTHER: Final = old_option.Some(TEST_VALUE)
RUST_SOME_DIFF: Final = pc.Some(99)
PYTHON_SOME_DIFF: Final = old_option.Some(99)


def bench_rust_eq_same() -> bool:
    """Rust == check (same value)."""
    return RUST_SOME == RUST_SOME_OTHER


def bench_python_eq_same() -> bool:
    """Python == check (same value)."""
    return PYTHON_SOME == PYTHON_SOME_OTHER


def bench_rust_eq_method_same() -> bool:
    """Rust .eq() method (same value) - avoids isinstance."""
    return RUST_SOME.eq(RUST_SOME_OTHER)


def bench_python_eq_method_same() -> bool:
    """Python .eq() method (same value) - uses is_some() instead of isinstance."""
    return PYTHON_SOME.eq(PYTHON_SOME_OTHER)


def bench_rust_eq_diff() -> bool:
    """Rust == check (different value)."""
    return RUST_SOME == RUST_SOME_DIFF


def bench_python_eq_diff() -> bool:
    """Python == check (different value)."""
    return PYTHON_SOME == PYTHON_SOME_DIFF


# =============================================================================
# 4. MAP WITH CLOSURES
# =============================================================================


def _identity[T](x: T) -> T:
    return x


def _add_ten(x: int) -> int:
    return x + 10


def bench_rust_map_identity() -> object:
    """Rust map with identity closure."""
    return RUST_SOME.map(_identity)


def bench_python_map_identity() -> object:
    """Python map with identity closure."""
    return PYTHON_SOME.map(_identity)


def bench_rust_map_lambda() -> object:
    """Rust map with inline lambda."""
    return RUST_SOME.map(lambda x: x + 10)


def bench_python_map_lambda() -> object:
    """Python map with inline lambda."""
    return PYTHON_SOME.map(lambda x: x + 10)


def bench_rust_map_function() -> object:
    """Rust map with named function."""
    return RUST_SOME.map(_add_ten)


def bench_python_map_function() -> object:
    """Python map with named function."""
    return PYTHON_SOME.map(_add_ten)


# =============================================================================
# 5. AND_THEN (flatmap)
# =============================================================================


def _wrap_add_ten(x: int) -> pc.Some[int]:
    return pc.Some(x + 10)


def _wrap_add_ten_py(x: int) -> old_option.Some[int]:
    return old_option.Some(x + 10)


def bench_rust_and_then() -> object:
    """Rust and_then with wrapping function."""
    return RUST_SOME.and_then(_wrap_add_ten)


def bench_python_and_then() -> object:
    """Python and_then with wrapping function."""
    return PYTHON_SOME.and_then(_wrap_add_ten_py)


# =============================================================================
# 6. UNWRAP OPERATIONS
# =============================================================================


def bench_rust_unwrap() -> int:
    """Rust unwrap on Some."""
    return RUST_SOME.unwrap()


def bench_python_unwrap() -> int:
    """Python unwrap on Some."""
    return PYTHON_SOME.unwrap()


def bench_rust_unwrap_or() -> int:
    """Rust unwrap_or on Some."""
    return RUST_SOME.unwrap_or(0)


def bench_python_unwrap_or() -> int:
    """Python unwrap_or on Some."""
    return PYTHON_SOME.unwrap_or(0)


def bench_rust_unwrap_or_none() -> int:
    """Rust unwrap_or on None."""
    return RUST_NONE.unwrap_or(0)


def bench_python_unwrap_or_none() -> int:
    """Python unwrap_or on None."""
    return PYTHON_NONE.unwrap_or(0)


# =============================================================================
# 7. INTO (Pipeable trait)
# =============================================================================


def process_option(opt: object) -> str:
    """Process an Option, returns repr."""
    return repr(opt)


def noop(opt: object) -> object:
    """No-op function, just returns the argument."""
    return opt


def bench_rust_into() -> str:
    """Rust into with function."""
    return RUST_SOME.into(process_option)


def bench_python_into() -> str:
    """Python into with function."""
    return PYTHON_SOME.into(process_option)


def bench_rust_into_noop() -> object:
    """Rust into with noop (pure overhead test)."""
    return RUST_SOME.into(noop)


def bench_python_into_noop() -> object:
    """Python into with noop (pure overhead test)."""
    return PYTHON_SOME.into(noop)


def bench_rust_into_with_args() -> str:
    """Rust into with additional args."""
    return RUST_SOME.into(lambda opt, suffix: repr(opt) + suffix, "!")  # type: ignore[return-value]


def bench_python_into_with_args() -> str:
    """Python into with additional args."""
    return PYTHON_SOME.into(lambda opt, suffix: repr(opt) + suffix, "!")  # type: ignore[return-value]


# =============================================================================
# 8. CHAINED OPERATIONS
# =============================================================================


def bench_rust_chain() -> object:
    """Rust chained operations."""
    return (
        pc.Some(CHAIN_VALUE)
        .map(lambda x: x * 2)
        .filter(lambda x: x > CHAIN_THRESHOLD)
        .map(lambda x: x + 1)
    )


def bench_python_chain() -> object:
    """Python chained operations."""
    return (
        old_option.Some(CHAIN_VALUE)
        .map(lambda x: x * 2)
        .filter(lambda x: x > CHAIN_THRESHOLD)
        .map(lambda x: x + 1)
    )


# =============================================================================
# MAIN
# =============================================================================


def bench(
    category: str,
    name: str,
    rust_fn: Callable[[], object],
    python_fn: Callable[[], object],
) -> None:
    """Run a single benchmark multiple times and store median results."""
    rust_times = [timeit.timeit(rust_fn, number=N_RUNS) for _ in range(N_RUNS)]
    python_times = [timeit.timeit(python_fn, number=N_RUNS) for _ in range(N_RUNS)]
    rust_median = statistics.median(rust_times)
    python_median = statistics.median(python_times)
    speedup = python_median / rust_median
    RESULTS.append((category, name, rust_median, python_median, speedup))


def _run_all_benchmarks() -> None:
    # 1. Instantiation
    bench(
        "Instantiation", "Some(value)", bench_rust_some_direct, bench_python_some_direct
    )
    bench(
        "Instantiation",
        "Option(value)",
        bench_rust_option_dispatch_some,
        bench_python_option_dispatch_some,
    )
    bench(
        "Instantiation",
        "Option(None)",
        bench_rust_option_dispatch_none,
        bench_python_option_dispatch_none,
    )

    # 2. Type checking
    bench("Type Check", "is_some()", bench_rust_is_some, bench_python_is_some)
    bench("Type Check", "is_none()", bench_rust_is_none, bench_python_is_none)

    # 3. Equality
    bench("Equality", "== (same)", bench_rust_eq_same, bench_python_eq_same)
    bench(
        "Equality",
        ".eq() (same)",
        bench_rust_eq_method_same,
        bench_python_eq_method_same,
    )
    bench("Equality", "== (diff)", bench_rust_eq_diff, bench_python_eq_diff)

    # 4. Map
    bench("Map", "map(identity)", bench_rust_map_identity, bench_python_map_identity)
    bench("Map", "map(lambda)", bench_rust_map_lambda, bench_python_map_lambda)
    bench("Map", "map(function)", bench_rust_map_function, bench_python_map_function)

    # 5. And_then
    bench("Flatmap", "and_then()", bench_rust_and_then, bench_python_and_then)

    # 7. Into
    bench("Into (FFI)", "into(func)", bench_rust_into, bench_python_into)
    bench("Into (FFI)", "into(noop)", bench_rust_into_noop, bench_python_into_noop)
    bench(
        "Into (FFI)",
        "into(func, arg)",
        bench_rust_into_with_args,
        bench_python_into_with_args,
    )
    # 8. Chain
    bench("Chain", "map.filter.map", bench_rust_chain, bench_python_chain)


def _display_results() -> None:
    table = Table(
        title=f"Option Benchmarks ({N_RUNS:,} iterations, median of {N_REPEATS} runs)"
    )
    table.add_column("Category", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Rust (s, median)", justify="right", style="green")
    table.add_column("Python (s, median)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right")

    for category, name, rust_time, python_time, speedup in RESULTS:
        speedup_style = "green bold" if speedup > 1 else "red bold"
        speedup_str = Text(f"{speedup:.2f}x", style=speedup_style)
        table.add_row(
            category, name, f"{rust_time:.4f}", f"{python_time:.4f}", speedup_str
        )

    CONSOLE.print(table)

    # Summary
    median_speedup = statistics.median([r[4] for r in RESULTS])
    wins = sum(1 for r in RESULTS if r[4] > 1)
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
    n_focused_runs = 2000
    n_focused_repeats = 2000
    old_name = old.func.__name__
    new_name = new.func.__name__
    CONSOLE.print(Text("Running Focused Robustness Benchmark...", style="bold blue"))
    CONSOLE.print(
        Text(
            f"{n_focused_runs:,} iterations x {n_focused_repeats:,} repeats for statistical significance...",
            style="dim",
        )
    )
    CONSOLE.print()

    # Benchmark new
    new_times = [
        timeit.timeit(new, number=n_focused_runs) for _ in range(n_focused_repeats)
    ]

    # Benchmark old
    old_times = [
        timeit.timeit(
            old,
            number=n_focused_runs,
        )
        for _ in range(n_focused_repeats)
    ]

    # Benchmark eq_direct (FFI direct access, no type checking)
    new_median = statistics.median(new_times)
    new_mean = statistics.mean(new_times)
    new_stddev = statistics.stdev(new_times)

    old_median = statistics.median(old_times)
    old_mean = statistics.mean(old_times)
    old_stddev = statistics.stdev(old_times)

    # Calculate relative metrics
    speedup = (
        old_median / new_median
    )  # If > 1: eq_old is faster; if < 1: __eq__ is faster
    improvement_pct = ((old_median - new_median) / old_median) * 100
    relative_stddev_new = (new_stddev / new_median) * 100
    relative_stddev_old = (old_stddev / old_median) * 100

    # Quartiles
    new_q1 = sorted(new_times)[len(new_times) // 4]
    new_q3 = sorted(new_times)[3 * len(new_times) // 4]
    old_q1 = sorted(old_times)[len(old_times) // 4]
    old_q3 = sorted(old_times)[3 * len(old_times) // 4]

    table = Table(
        title=f"{new_name} vs {old_name} (isinstance)\n{n_focused_runs:,} ops x {n_focused_repeats} repeats"
    )
    table.add_column("Metric", style="cyan")
    table.add_column("new", justify="right", style="green")
    table.add_column("old", justify="right", style="yellow")
    table.add_column("Relative", justify="right", style="magenta")

    if speedup > 1:
        speedup_msg = Text("new ", style="green bold") + Text(
            f"{speedup:.2f}x faster", style="green bold"
        )
    else:
        speedup_msg = Text("old ", style="yellow bold") + Text(
            f"{1 / speedup:.2f}x faster", style="yellow bold"
        )

    table.add_row("Speedup", "1.00x", f"{1 / speedup:.3f}x", speedup_msg)

    if improvement_pct > 0:
        improvement_label = Text("faster", style="green bold")
    else:
        improvement_label = Text("slower", style="yellow bold")
    improvement_text = (
        Text(f"{improvement_pct:+.1f}% ", style="dim") + improvement_label
    )
    table.add_row("Improvement", "—", "—", improvement_text)

    # Median
    table.add_row(
        "Median (rel)",
        "1.00",
        f"{old_median / new_median:.3f}",
        f"{improvement_pct:+.1f}%",
    )

    # Mean
    table.add_row(
        "Mean (rel)",
        "1.00",
        f"{old_mean / new_mean:.3f}",
        f"{((old_mean - new_mean) / old_mean * 100):+.1f}%",
    )

    # Variability (CV%)
    table.add_row(
        "Variability (CV%)",
        f"{relative_stddev_new:.2f}%",
        f"{relative_stddev_old:.2f}%",
        f"{relative_stddev_old - relative_stddev_new:+.2f}%",
    )

    # IQR
    table.add_row(
        "IQR (rel)",
        f"{(new_q3 - new_q1) / new_median:.4f}",
        f"{(old_q3 - old_q1) / old_median:.4f}",
        f"{((old_q3 - old_q1) / old_median - (new_q3 - new_q1) / new_median):+.4f}",
    )

    CONSOLE.print(table)
    CONSOLE.print()
    if speedup > 1:
        CONSOLE.print(
            Text(
                f"✓ {new_name} is {speedup:.2f}x faster ({improvement_pct:.1f}% improvement)",
                style="bold green",
            )
        )
    else:
        CONSOLE.print(
            Text(
                f"✗ {old_name} is {1 / speedup:.2f}x faster ({abs(improvement_pct):.1f}% regression)",
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
    _run_focused_benchmark(
        old=partial(RUST_SOME.eq, RUST_SOME_OTHER),
        new=partial(RUST_SOME.eq_test, RUST_SOME_OTHER),  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    app()
