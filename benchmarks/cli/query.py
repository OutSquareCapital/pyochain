from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, Final

import polars as pl

from pyochain import Iter

from .._utils import Lib, PlEnum
from ._common import GET_PATH

if TYPE_CHECKING:
    from collections.abc import Sequence

SEC_TO_MS_FACTOR: Final[int] = 1_000_000
"""Convert seconds to microseconds."""


class Cols(PlEnum):
    """Column names used in the benchmarks."""

    Size = auto()
    TimeStamp = auto()
    Relative = auto()
    Lib = auto()
    Commit = auto()
    Run = auto()
    Param = auto()
    Min = auto()
    Name = auto()


GROUP = (Cols.Param, Cols.Size)


def run(method: str, agg_by_commit: bool) -> pl.DataFrame:
    param = Cols.Param.pl().list
    return (
        read_jsons(True)
        .filter(Cols.Name.pl().eq(method))
        .select(
            Cols.TimeStamp,
            param.first().cast(pl.String).alias(Cols.Size),
            param.get(1).cast(Lib).alias(Cols.Lib),
            # Additional params besides size and library
            param.slice(2, param.len() - 2).alias(Cols.Param),
            pl.col("stats").struct.field(Cols.Min),
            Cols.Commit,
        )
        .pivot(
            Cols.Lib,
            (Lib.SortedContainers, Lib.Pyochain),
            index=(Cols.TimeStamp, *GROUP, Cols.Commit),
        )
        .with_columns(
            Lib.SortedContainers
            .pl()
            .median()
            .over(GROUP)
            .truediv(Lib.Pyochain)
            .round(3)
            .alias(Cols.Relative),
            Lib.Pyochain.pl().mul(SEC_TO_MS_FACTOR),
            Lib.SortedContainers.pl().mul(SEC_TO_MS_FACTOR),
        )
        .sort(Cols.TimeStamp, Cols.Size)
        .pipe(maybe_agg_by_commit, agg_by_commit)
        .collect()
    )


def read_jsons(unnest: bool) -> pl.LazyFrame:
    cols = _selected_cols() if unnest else pl.all()
    return (
        Iter(GET_PATH.glob("*.json"))
        .map(lambda path: pl.read_json(path).lazy().select(cols))
        .collect(pl.concat)
    )


def maybe_agg_by_commit(lf: pl.LazyFrame, agg_by_commit: bool) -> pl.LazyFrame:
    if agg_by_commit:
        return lf.group_by(Cols.Commit, *GROUP, maintain_order=True).agg(
            Cols.TimeStamp.pl().first(),
            pl.selectors.numeric().median().name.keep(),
        )
    else:
        return lf.with_columns(
            Cols.TimeStamp.pl().cum_count().over(GROUP).alias(Cols.Run)
        )


def _selected_cols() -> Sequence[pl.Expr]:
    benchmark = pl.col("benchmarks").list.explode().struct.field
    return (
        benchmark("fullname")
        .str.split("/")
        .list.first()
        .str.split("[")
        .list.first()
        .alias(Cols.Name),
        benchmark(Cols.Param).str.split("-"),
        benchmark("stats"),
        pl.col("commit_info").struct.field("id").alias(Cols.Commit),
        pl
        .col("datetime")
        .str.to_datetime(time_unit="ms", time_zone="UTC")
        .alias(Cols.TimeStamp),
    )
