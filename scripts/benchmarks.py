"""Benchmarks: Pyochain vs pure Python."""

from __future__ import annotations

import timeit
from collections.abc import Callable
from functools import wraps

from rich.console import Console
from rich.table import Table

import pyochain as pc


def _process(x: int, multiplier: int, offset: int) -> int:
    return x * multiplier + offset


def _outer_fn(x: int, factor: int) -> int:
    return x * factor


def _inner_fn(x: int, offset: int) -> int:
    return x + offset


# ============================================================================
# Benchmarks Registry
# ============================================================================

BENCHMARKS = pc.Dict[str, Callable[[], pc.Dict[str, float]]].new()


def benchmark(
    name: str,
) -> Callable[[Callable[[], pc.Dict[str, float]]], Callable[[], pc.Dict[str, float]]]:
    """Decorator to register a benchmark."""

    def decorator(
        func: Callable[[], pc.Dict[str, float]],
    ) -> Callable[[], pc.Dict[str, float]]:
        @wraps(func)
        def wrapper() -> pc.Dict[str, float]:
            return func()

        BENCHMARKS[name] = wrapper
        return wrapper

    return decorator


@benchmark("Simple Map (1K items, 10K iter)")
def bench_simple_map() -> pc.Dict[str, float]:
    """Simple map avec transformation."""
    data = pc.Iter(range(1000)).into(list)
    mult, off = 3, 10
    nb = 10_000

    def _transform(x: int) -> int:
        return _process(x, mult, off)

    return pc.Dict.from_kwargs(
        python=timeit.timeit(
            lambda: tuple(_process(x, mult, off) for x in data), number=nb
        ),
        pyochain_lambda=timeit.timeit(
            lambda: pc.Iter(data).map(lambda x: _process(x, mult, off)).collect(),
            number=nb,
        ),
        pyochain_closure=timeit.timeit(
            lambda: pc.Iter(data).map(_transform).collect(), number=nb
        ),
    )


@benchmark("Nested Map (100x10 items, 1K iter)")
def bench_nested_map() -> pc.Dict[str, float]:
    """Nested map (pire cas)."""
    data = [list(range(10)) for _ in range(100)]
    factor, offset = 2, 10
    nb = 1000

    def _transform_nested(outer: list[int]) -> pc.Seq[int]:
        return (
            pc.Iter(outer)
            .map(lambda x: _inner_fn(_outer_fn(x, factor), offset))
            .collect()
        )

    return pc.Dict.from_kwargs(
        python=timeit.timeit(
            lambda: tuple(
                tuple(_inner_fn(_outer_fn(x, factor), offset) for x in outer)
                for outer in data
            ),
            number=nb,
        ),
        pyochain_lambda=timeit.timeit(
            lambda: pc.Iter(data)
            .map(
                lambda outer: pc.Iter(outer)
                .map(lambda x: _outer_fn(x, factor))
                .map(lambda x: _inner_fn(x, offset))
                .collect()
            )
            .collect(),
            number=nb,
        ),
        pyochain_closure=timeit.timeit(
            lambda: pc.Iter(data).map(_transform_nested).collect(),
            number=nb,
        ),
        pyochain_flatten=timeit.timeit(
            lambda: pc.Iter(data)
            .flat_map(
                lambda outer: pc.Iter(outer)
                .map(lambda x: _outer_fn(x, factor))
                .map(lambda x: _inner_fn(x, offset))
            )
            .collect(),
            number=nb,
        ),
    )


# ============================================================================
# Runner
# ============================================================================
console = Console()


def run_all_benchmarks() -> pc.Dict[str, pc.Dict[str, int | float]]:
    """Exécute tous les benchmarks."""
    console.print("\n🔬 [bold cyan]PYOCHAIN vs PYTHON[/bold cyan]")
    console.print("=" * 60)

    def _run(
        item: tuple[str, Callable[[], pc.Dict[str, float]]],
    ) -> tuple[str, pc.Dict[str, float]]:
        return (item[0], item[1]())

    return pc.Iter(BENCHMARKS.items()).map(_run).into(pc.Dict)


def _summary(results: pc.Dict[str, pc.Dict[str, int | float]]) -> None:
    # Afficher chaque benchmark
    results.iter().for_each(
        lambda item: _display_benchmark(console, item.key, item.value)
    )

    # Summary
    console.print(f"\n{'=' * 60}")
    console.print("[bold yellow]OVERHEAD PYOCHAIN[/bold yellow]")
    console.print(f"{'=' * 60}\n")

    results.iter().filter(lambda item: "pyochain_lambda" in item.value).for_each(
        lambda item: _display_overhead(console, item.key, item.value)
    )

    console.print(f"\n{'=' * 60}\n")


def _display_benchmark(console: Console, name: str, r: pc.Dict[str, float]) -> None:
    """Affiche les résultats d'un benchmark."""
    baseline = r["python"]

    table = Table(title=name, show_header=True, header_style="bold magenta")
    table.add_column("Approche", style="cyan")
    table.add_column("Temps (s)", justify="right", style="green")
    table.add_column("vs Python", justify="right", style="yellow")

    r.iter().sort(lambda x: x.value).iter().for_each(
        lambda item: table.add_row(
            item.key, f"{item.value:.4f}", f"{item.value / baseline:.2f}x"
        )
    )

    console.print(table)


def _display_overhead(console: Console, name: str, r: pc.Dict[str, float]) -> None:
    """Affiche l'overhead pour un benchmark."""
    overhead = (r["pyochain_lambda"] / r["python"] - 1) * 100
    color = _get_color(overhead)
    console.print(f"{name:<40} [bold {color}]{overhead:+.1f}%[/bold {color}]")


def _get_color(
    overhead: float, red_treshold: float = 50.0, yellow_treshold: float = 20.0
) -> str:
    return (
        "red"
        if overhead > red_treshold
        else "yellow"
        if overhead > yellow_treshold
        else "green"
    )


def main() -> None:
    """Entry point."""
    _summary(run_all_benchmarks())


if __name__ == "__main__":
    main()
