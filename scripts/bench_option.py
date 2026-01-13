"""Comprehensive benchmarks for Option types: Rust vs Python implementations."""

import statistics
import timeit
from collections.abc import Callable
from typing import Final

from rich.console import Console
from rich.table import Table

import pyochain as pc
from pyochain import old_option

N_RUNS: Final[int] = 100_000
TEST_VALUE: Final[int] = 42
CHAIN_VALUE: Final[int] = 5
CHAIN_THRESHOLD: Final[int] = 5


CONSOLE: Final = Console()
# Store all runs for each benchmark, then compute median
RESULTS: list[tuple[str, str, float, float, float]] = []
N_REPEATS: Final[int] = 7  # Number of times to repeat each benchmark for median


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
    rust_times = [timeit.timeit(rust_fn, number=N_RUNS) for _ in range(N_REPEATS)]
    python_times = [timeit.timeit(python_fn, number=N_RUNS) for _ in range(N_REPEATS)]
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
        speedup_str = f"[{speedup_style}]{speedup:.2f}x[/{speedup_style}]"
        table.add_row(
            category, name, f"{rust_time:.4f}", f"{python_time:.4f}", speedup_str
        )

    CONSOLE.print(table)

    # Summary
    median_speedup = statistics.median([r[4] for r in RESULTS])
    wins = sum(1 for r in RESULTS if r[4] > 1)
    CONSOLE.print(f"\n[bold]Median speedup:[/bold] {median_speedup:.2f}x")
    CONSOLE.print(f"[bold]Rust wins:[/bold] {wins}/{len(RESULTS)}")


def main() -> None:
    """Run all benchmarks and display results."""
    CONSOLE.print("[bold blue]Running Option benchmarks...[/bold blue]\n")
    _run_all_benchmarks()
    _display_results()


if __name__ == "__main__":
    main()
