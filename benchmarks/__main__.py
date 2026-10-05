"""CLI entry point for benchmark scripts."""

from __future__ import annotations

from typing import Annotated

import typer as tp

app = tp.Typer(name="bench", help="Run pytest-benchmark and analyze the results.")

MethodArg = Annotated[str, tp.Argument(help="Benchmark method name")]
GroupByCommitArg = Annotated[bool, tp.Option(help="Group results by commit.")]


@app.command()
def run(
    method: MethodArg,
    repeat: Annotated[int, tp.Option(help="How many times the benchmark is run")] = 1,
    *,
    calibrate: Annotated[
        bool,
        tp.Option(
            help="Run original sortedcontainers with pyochain for relative comparison"
        ),
    ] = False,
) -> None:
    """Run a benchmark x time.

    The relative speed is compared to a median value across runs for `sortedcontainers`.

    If you don't already have saved results for it, make sure to use the `--calibrate` option.
    """
    from .cli import bench_run, check_method

    return (
        check_method(method)
        .and_then(lambda m: bench_run.run(m, repeat, calibrate=calibrate))
        .unwrap()
    )


@app.command()
def plot(method: MethodArg, *, group_by_commit: GroupByCommitArg = False) -> None:
    """Plot the benchmark results with `plotly` for a given method."""
    from .cli import check_method, display, query

    df = check_method(method).map(query.run, group_by_commit).unwrap()
    x_axis = query.Cols.Commit if group_by_commit else query.Cols.Run
    display.absolute(df, method, x_axis)
    display.relative(df, method, x_axis)


@app.command()
def show(
    method: MethodArg,
    *,
    raw: Annotated[
        bool,
        tp.Option(help="Render the table as raw markdown instead of pretty printing"),
    ] = False,
) -> None:
    """Show the benchmark results in the terminal for a given method."""

    from .cli import check_method, display, query

    return (
        check_method(method)
        .map(query.run, False)
        .map(display.terminal, raw=raw)
        .unwrap()
    )


if __name__ == "__main__":
    app()
