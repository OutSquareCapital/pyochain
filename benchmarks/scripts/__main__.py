"""CLI entry point for benchmark scripts."""

from __future__ import annotations

from typing import Annotated

import typer as tp

app = tp.Typer(name="bench", help="Run pytest-benchmark and analyze the results.")

MethodArg = Annotated[str, tp.Argument(help="Benchmark method name")]


@app.command()
def run(
    method: MethodArg,
    repeat: Annotated[int, tp.Option(help="How many time the benchmark is run")] = 1,
) -> None:
    """Run a benchmark x time."""
    from . import bench_run
    from ._common import check_method

    return check_method(method).and_then(lambda m: bench_run.run(m, repeat)).unwrap()


@app.command()
def plot(
    method: MethodArg,
    *,
    plot: Annotated[bool, tp.Option(help="Display interactive plots.")] = False,
    show: Annotated[bool, tp.Option(help="Show tabular results in console.")] = False,
) -> None:
    """Compute ratios and render the plots for the selected group, compared to sortedcontainers."""
    from . import bench_plots
    from ._common import check_method

    return (
        check_method(method)
        .map(lambda m: bench_plots.main(m, plot=plot, show=show))
        .unwrap()
    )


if __name__ == "__main__":
    app()
