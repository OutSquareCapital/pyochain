"""Benchmark plotting script."""

import platform
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Final

import plotly.express as px
import polars as pl

from pyochain import Iter

PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
PATH: Final[Path] = Path(".benchmarks", "sortedlist", PLATFORM_DIR)
"""Path to the benchmark results directory."""


class Lib(StrEnum):
    """Libraries used in the benchmarks."""

    Pyochain = auto()
    SortedContainers = auto()


def main(group: str) -> None:
    """Read benchmark data for one method, compute ratios, and generate plots."""
    df = _get_df(group)
    ratios = _get_ratios(df)
    df.show(-1)
    ratios.show(-1)
    _absolute_plot(df, group)
    _relative_plot(ratios, group)


def _absolute_plot(df: pl.DataFrame, group: str) -> None:
    return px.bar(  # pyright: ignore[reportUnknownMemberType]
        df,
        title=f"{group}: pyochain vs sortedcontainers across runs",
        x="run",
        y="median",
        color="lib",
        facet_col="size",
        barmode="group",
        log_y=True,
        template="plotly_dark",
    ).show()


def _relative_plot(ratios: pl.DataFrame, group: str) -> None:
    return (
        px
        .bar(  # pyright: ignore[reportUnknownMemberType]
            ratios,
            title=f"{group}: speedup of pyochain vs sortedcontainers across runs",
            x="run",
            y="speedup",
            barmode="group",
            color="size",
            template="plotly_dark",
        )
        .add_hline(y=1)
        .show()
    )


def _get_df(group: str) -> pl.DataFrame:
    benchmark = pl.col("benchmarks").list.explode().struct.field
    stat = benchmark("stats").struct.field
    param = pl.col("param").str.split("-").list
    selected_cols = (
        benchmark("fullname"),
        benchmark("name"),
        benchmark("param"),
        stat("min"),
        stat("max"),
        stat("median"),
        stat("stddev"),
        stat("total"),
    )
    return (
        Iter(PATH.glob("*.json"))
        .sort_by(lambda path: path.stat().st_mtime)
        .iter()
        .map(
            lambda path: (
                pl
                .read_json(path)
                .lazy()
                .select(selected_cols)
                .with_columns(pl.lit(path.stem.split("_")[0]).alias("run"))
            )
        )
        .collect(pl.concat)
        .with_columns(
            pl
            .col("name")
            .str.split("[")
            .list.first()
            .str.strip_prefix("test_")
            .alias("method"),
            param.first().cast(pl.UInt32()).alias("size"),
            param.last().cast(Lib).alias("lib"),
        )
        .filter(pl.col("method") == group)
        .sort("size", "run", "lib")
        .collect()
    )


def _get_ratios(df: pl.DataFrame) -> pl.DataFrame:
    idx_cols = ("method", "run", "size")
    return (
        df
        .select(*idx_cols, "median", "lib")
        .pivot("lib", index=idx_cols)
        .select(
            "run",
            pl.col("size").cast(pl.String()),
            "method",
            pl
            .col(Lib.SortedContainers)
            .truediv(Lib.Pyochain)
            .round(3)
            .alias("speedup"),
        )
    )


if __name__ == "__main__":
    main(sys.argv[1])
