"""Benchmark plotting script."""

from enum import StrEnum, auto
from pathlib import Path

import plotly.express as px
import polars as pl

BENCHS = Path(".benchmarks/Windows-CPython-3.13-64bit")
PATH = BENCHS.joinpath("0071_sortedlist.json")


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
    ).show()


def _get_df() -> pl.DataFrame:
    stat = pl.col("stats").struct.field
    param = pl.col("param").str.split("-").list
    return (
        pl
        .read_json(PATH)
        .lazy()
        .select(pl.col("benchmarks").list.explode().struct.unnest())
        .select(
            pl
            .col("name")
            .str.split("[")
            .list.first()
            .str.strip_prefix("test_")
            .alias("method"),
            param.first().cast(pl.UInt32()).alias("size"),
            param.last().cast(Lib).alias("lib"),
            stat("min"),
            stat("max"),
            stat("median"),
            stat("stddev"),
            stat("total"),
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
            pl.col(Lib.SORTEDCONTAINERS).truediv(Lib.PYOCHAIN).sub(1).alias("speedup"),
        )
    )


if __name__ == "__main__":
    main()
