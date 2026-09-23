"""Benchmark plotting script."""

import platform
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Final

import plotly.express as px
import polars as pl

from pyochain import Iter

# TODO: 0071_sortedlist.json is the first time we ran the benchmarks.
# We should use it as a third reference to ensure no regressions.

PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
PATH: Final[Path] = Path(".benchmarks", "sortedlist", PLATFORM_DIR)
"""Path to the benchmark results directory."""


class Lib(StrEnum):
    """Libraries used in the benchmarks."""

    Pyochain = auto()
    SortedContainers = auto()


def main() -> None:
    """Read benchmark data, compute ratios, and generate plots."""
    df = _get_df()
    df.show(-1)
    ratios = _get_ratios(df)
    ratios.show(-1)
    px.bar(  # pyright: ignore[reportUnknownMemberType]
        df,
        title="Benchmark results for pyochain vs sortedcontainers",
        x="test",
        y="median",
        color="lib",
        barmode="group",
        log_y=True,
        template="plotly_dark",
    ).show()
    px.bar(  # pyright: ignore[reportUnknownMemberType]
        ratios,
        title="Speedup of pyochain vs sortedcontainers",
        x="test",
        y="speedup",
        barmode="relative",
        color="method",
        template="plotly_dark",
    ).add_hline(y=1).show()


def _get_df() -> pl.DataFrame:
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
        .map(pl.read_json)
        .map(pl.DataFrame.lazy)
        .map(lambda df: df.select(selected_cols))
        .collect(pl.concat)
        .unique("fullname", keep="last")
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
        .with_columns(
            pl
            .col("method")
            .add("-")
            .add(pl.col("size").cast(pl.String()))
            .alias("test")
        )
        .sort("method", "size", "lib")
        .collect()
    )


def _get_ratios(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df
        .select("method", "median", "lib", "test")
        .pivot("lib", index=("method", "test"))
        .select(
            "test",
            "method",
            pl
            .col(Lib.SortedContainers)
            .truediv(Lib.Pyochain)
            .round(3)
            .alias("speedup"),
        )
    )


if __name__ == "__main__":
    main()
