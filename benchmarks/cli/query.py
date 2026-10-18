from __future__ import annotations

from enum import auto
from typing import TYPE_CHECKING, Final

import polars as pl

from pyochain import Iter

from ._common import GET_PATH, PREFIX, Lib, PlEnum

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


def run(method: str, agg_by_commit: bool) -> pl.DataFrame:
    param = Cols.Param.pl().str.split("-").list
    cols = _selected_cols()
    return (
        Iter(GET_PATH.glob("*.json"))
        .map(lambda path: pl.read_json(path).lazy().select(cols))
        .collect(pl.concat)
        .filter(
            Cols.Name
            .pl()
            .str.split("[")
            .list.first()
            .str.strip_prefix(PREFIX)
            .eq(method)
        )
        .select(
            Cols.TimeStamp,
            param.first().cast(pl.String).alias(Cols.Size),
            # Additional params besides size and library
            param.slice(1, param.len() - 2).alias(Cols.Param),
            Cols.Min,
            param.last().cast(Lib).alias(Cols.Lib),
            Cols.Commit,
        )
        .pivot(
            Cols.Lib,
            (Lib.SortedContainers, Lib.Pyochain),
            index=(Cols.TimeStamp, Cols.Param, Cols.Size, Cols.Commit),
        )
        .with_columns(
            Lib.SortedContainers
            .pl()
            .median()
            .over(Cols.Param, Cols.Size)
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


def maybe_agg_by_commit(lf: pl.LazyFrame, agg_by_commit: bool) -> pl.LazyFrame:
    if agg_by_commit:
        return lf.group_by(Cols.Commit, Cols.Param, Cols.Size, maintain_order=True).agg(
            Cols.TimeStamp.pl().first(),
            pl.selectors.numeric().median().name.keep(),
        )
    else:
        return lf.with_columns(
            Cols.TimeStamp.pl().cum_count().over(Cols.Param, Cols.Size).alias(Cols.Run)
        )


def _selected_cols() -> Sequence[pl.Expr]:
    commit_infos = pl.col("commit_info").struct.field
    benchmark = pl.col("benchmarks").list.explode().struct.field
    stat = benchmark("stats").struct.field
    return (
        benchmark("fullname"),
        benchmark(Cols.Name),
        benchmark(Cols.Param),
        stat(Cols.Min),
        stat("max"),
        stat("median"),
        stat("stddev"),
        stat("total"),
        commit_infos("id").alias(Cols.Commit),
        pl
        .col("datetime")
        .str.to_datetime(time_unit="ms", time_zone="UTC")
        .alias(Cols.TimeStamp),
    )
