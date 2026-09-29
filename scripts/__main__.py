"""CLI entry point for benchmark plotting."""

from __future__ import annotations

from typing import Annotated

import typer as tp

app = tp.Typer(name="bench-plots", help="Plot and tabulate pytest-benchmark results.")


@app.command()
def plot(
    method: Annotated[str, tp.Argument(help="Benchmark method name, e.g. 'iter'.")],
    *,
    plot: Annotated[bool, tp.Option(help="Display interactive plots.")] = False,
) -> None:
    """Compute ratios and render the plots for the selected group, compared to sortedcontainers."""
    from . import bench_plots

    bench_plots.main(method, plot=plot)


if __name__ == "__main__":
    app()
