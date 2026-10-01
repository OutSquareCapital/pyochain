"""Benchmark plotting script."""

from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, Final

import plotly.express as px
import polars as pl

from pyochain import Iter

from .._utils import SIZES
from ._common import GET_PATH, PREFIX, Lib, Method, PlEnum

if TYPE_CHECKING:
    from collections.abc import Sequence

Sizes: Final[pl.Enum] = SIZES.iter().map(str).collect(pl.Enum)


class Cols(PlEnum):
    """Column names used in the benchmarks."""

    Size = auto()
    Run = auto()
    Relative = auto()
    Lib = auto()
    Commit = auto()


def main(method: Method, *, plot: bool, show: bool) -> None:
    """Read benchmark data for one method, compute ratios, and generate plots."""
    df = _get_df(method)
    if show:
        _ = pl.Config().set_tbl_hide_column_data_types(True)
        df.show(-1)
    if plot:
        _absolute_plot(df, method)
        _relative_plot(df, method)


def _absolute_plot(df: pl.DataFrame, method: str) -> None:
    return px.line(  # pyright: ignore[reportUnknownMemberType]
        df,
        title=f"{method}: absolute speed across runs",
        x=Cols.Commit,
        y=Lib.Pyochain,
        color=Cols.Size,
        log_y=True,
        template="plotly_dark",
    ).show()


def _relative_plot(df: pl.DataFrame, method: str) -> None:
    return (
        px
        .line(  # pyright: ignore[reportUnknownMemberType]
            df,
            title=f"{method}: speedup of pyochain vs sortedcontainers across runs",
            x=Cols.Commit,
            y=Cols.Relative,
            color=Cols.Size,
            template="plotly_dark",
        )
        .add_hline(y=1)
        .show()
    )


def _get_df(method: str) -> pl.DataFrame:
    param = pl.col("param").str.split("-").list
    cols = _selected_cols()
    return (
        Iter(GET_PATH.glob("*.json"))
        .enumerate()
        .map_star(
            lambda idx, path: (
                pl
                .read_json(path)
                .lazy()
                .select(cols)
                .with_columns(pl.lit(idx).alias(Cols.Run))
            )
        )
        .collect(pl.concat)
        .filter(
            pl
            .col("name")
            .str.split("[")
            .list.first()
            .str.strip_prefix(PREFIX)
            .eq(method)
        )
        .select(
            Cols.Run,
            param.first().cast(Sizes).alias(Cols.Size),
            "min",
            param.last().cast(Lib).alias(Cols.Lib),
            Cols.Commit,
        )
        .pivot(
            Cols.Lib,
            (Lib.SortedContainers, Lib.Pyochain),
            index=(Cols.Run, Cols.Size, Cols.Commit),
        )
        .with_columns(
            Lib.SortedContainers
            .pl()
            .median()
            .over(Cols.Size)
            .truediv(Lib.Pyochain)
            .round(3)
            .alias(Cols.Relative),
        )
        .sort(Cols.Run, Cols.Size)
        .group_by(Cols.Commit, Cols.Size, maintain_order=True)
        .agg(pl.selectors.numeric().median().name.keep())
        .collect()
    )


def _selected_cols() -> Sequence[pl.Expr]:
    commit_infos = pl.col("commit_info").struct.field
    benchmark = pl.col("benchmarks").list.explode().struct.field
    stat = benchmark("stats").struct.field
    return (
        benchmark("fullname"),
        benchmark("name"),
        benchmark("param"),
        stat("min"),
        stat("max"),
        stat("median"),
        stat("stddev"),
        stat("total"),
        Cols.Run.pl(),
        commit_infos("id").alias(Cols.Commit),
    )
