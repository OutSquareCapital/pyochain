"""Benchmarks for pyochain developments."""

import subprocess
from datetime import UTC, datetime

import framelib as fl
import polars as pl

import pyochain as pc

from ._registery import BENCHMARKS, Row, collect_raw_timings


class BenchmarksSchema(fl.Schema):
    """Schema for aggregated benchmark median results."""

    id = fl.String(primary_key=True)
    category = fl.String()
    name = fl.String()
    size = fl.UInt32()
    timestamp = fl.Datetime()
    git_hash = fl.String()
    median = fl.Float64()
    runs = fl.UInt32()


class BenchDb(fl.DataBase):
    """DuckDB database for storing benchmark results."""

    results = fl.Table(BenchmarksSchema)


class Benchmarks(fl.Folder):
    """Folder for storing benchmark databases."""

    db = BenchDb()


def run_pipeline() -> pl.DataFrame:
    """Persist aggregated benchmark results to DuckDB."""
    return (
        BENCHMARKS.ok_or("No benchmarks registered!")
        .map(collect_raw_timings)
        .map(_compute_all_stats)
        .and_then(_try_collect)
        .unwrap()
    )


def _try_collect(lf: pl.LazyFrame) -> pc.Result[pl.DataFrame, str]:
    """Try to collect a LazyFrame, with error handling."""
    try:
        return pc.Ok(lf.collect())
    except (
        pl.exceptions.ColumnNotFoundError,
        pl.exceptions.InvalidOperationError,
    ) as e:
        return pc.Err(f"{e}")


def _get_git_hash() -> pc.Result[str, Exception]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
        )
        return pc.Ok(result.stdout.strip())
    except Exception as e:  # noqa: BLE001
        return pc.Err(e)


def _compute_all_stats(raw_rows: pc.Seq[Row]) -> pl.LazyFrame:
    """Compute median stats from raw timings, returns atomic rows ready for DB."""
    now = datetime.now(tz=UTC)
    return (
        pl.LazyFrame(
            raw_rows,
            schema=["category", "name", "size", "run_idx", "time"],
            orient="row",
        )
        .group_by("category", "name", "size")
        .agg(
            pl.col("time").median().alias("median"),
            pl.len().alias("runs"),
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("category"),
                    pl.col("name"),
                    pl.col("size"),
                    pl.lit(int(now.timestamp() * 1_000_000)),
                ],
                separator="-",
            ).alias("id"),
            pl.lit(now).alias("timestamp"),
            _get_git_hash()
            .map(pl.lit)
            .expect("Failed to get git hash")
            .alias("git_hash"),
        )
        .select(BenchDb.results.model.schema().keys())
    )
