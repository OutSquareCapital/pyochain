from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, Final

import polars as pl

from pyochain import Iter

from .._utils import SIZES
from ._common import GET_PATH, PREFIX, Lib, PlEnum

if TYPE_CHECKING:
    from collections.abc import Sequence

Sizes: Final[pl.Enum] = SIZES.iter().map(str).collect(pl.Enum)


class Cols(PlEnum):
    """Column names used in the benchmarks."""

    Size = auto()
    TimeStamp = auto()
    Relative = auto()
    Lib = auto()
    Commit = auto()


def run(method: str, agg_by_commit: bool) -> pl.DataFrame:
    param = pl.col("param").str.split("-").list
    cols = _selected_cols()
    return (
        Iter(GET_PATH.glob("*.json"))
        .map(lambda path: pl.read_json(path).lazy().select(cols))
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
            Cols.TimeStamp,
            param.first().cast(Sizes).alias(Cols.Size),
            "min",
            param.last().cast(Lib).alias(Cols.Lib),
            Cols.Commit,
        )
        .pivot(
            Cols.Lib,
            (Lib.SortedContainers, Lib.Pyochain),
            index=(Cols.TimeStamp, Cols.Size, Cols.Commit),
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
        .sort(Cols.TimeStamp, Cols.Size)
        .pipe(lambda lf: _group_by_commit(lf) if agg_by_commit else lf)
        .collect()
    )


def _group_by_commit(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.group_by(Cols.Commit, Cols.Size, maintain_order=True).agg(
        pl.selectors.numeric().median().name.keep()
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
        commit_infos("id").alias(Cols.Commit),
        pl.col("datetime").cast(pl.Datetime("ms")).alias(Cols.TimeStamp),
    )
