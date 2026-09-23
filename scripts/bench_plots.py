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


class Lib(StrEnum):
    """Libraries used in the benchmarks."""

    PYOCHAIN = auto()
    SORTEDCONTAINERS = auto()


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
    benchmark = pl.col("benchmarks").list.explode()
    stat = benchmark.struct.field("stats").struct.field
    param = pl.col("param").str.split("-").list
    selected_cols = (
        benchmark.struct.field("fullname"),
        benchmark.struct.field("name"),
        benchmark.struct.field("param"),
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
            .col(Lib.SORTEDCONTAINERS)
            .truediv(Lib.PYOCHAIN)
            .round(3)
            .alias("speedup"),
        )
    )


if __name__ == "__main__":
    main()
