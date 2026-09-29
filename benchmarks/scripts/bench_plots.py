"""Benchmark plotting script."""

import platform
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import Final

import plotly.express as px
import polars as pl
from rich.console import Console
from rich.text import Text

from pyochain import Iter, Vec

from .. import test_sorted
from .._utils import SIZES

CONSOLE = Console()
PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
PATH: Final[Path] = Path(".benchmarks", "sortedlist", PLATFORM_DIR)
"""Path to the benchmark results directory."""
Sizes: Final[pl.Enum] = SIZES.iter().map(str).collect(pl.Enum)
PREFIX = "test_"


class PlEnum(StrEnum):
    """Base class for enums that can be used as polars expressions."""

    def pl(self) -> pl.Expr:
        """Return a polars expression for the enum value."""
        return pl.col(self.value)


class Cols(PlEnum):
    """Column names used in the benchmarks."""

    Size = auto()
    Run = auto()
    Relative = auto()
    Lib = auto()


class Lib(PlEnum):
    """Libraries used in the benchmarks."""

    Pyochain = auto()
    SortedContainers = auto()


def main(method: str, *, plot: bool, show: bool) -> None:
    """Read benchmark data for one method, compute ratios, and generate plots."""
    method = _check_method(method)
    df = _get_df(method)
    if show:
        _ = pl.Config().set_tbl_hide_column_data_types(True)
        df.show(-1)
    if plot:
        _absolute_plot(df, method)
        _relative_plot(df, method)


def _check_method(method: str) -> str:
    available_methods = _get_test_funcs()
    if not available_methods.contains(method):
        methods = available_methods.iter().map(str).map(lambda m: " - " + m).join("\n")
        txt = (
            Text(
                f"Error: Method '{method}' not found in benchmark tests.\n",
                style="bold red",
            )
            .append("\nAvailable methods:\n", style="bold yellow")
            .append(methods, style="bold green")
        )
        CONSOLE.print(txt)
        sys.exit(1)
    else:
        return method


def _get_test_funcs() -> Vec[str]:
    return (
        Iter(test_sorted.__dict__)
        .filter(lambda name: name.startswith(PREFIX))
        .map(lambda name: name.removeprefix(PREFIX))
        .sort()
    )


def _absolute_plot(df: pl.DataFrame, method: str) -> None:
    return px.line(  # pyright: ignore[reportUnknownMemberType]
        df,
        title=f"{method}: absolute speed across runs",
        x=Cols.Run,
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
            x=Cols.Run,
            y=Cols.Relative,
            color=Cols.Size,
            template="plotly_dark",
        )
        .add_hline(y=1)
        .show()
    )


def _get_df(method: str) -> pl.DataFrame:
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
                .select(*selected_cols, pl.lit(path.stem.split("_")[0]).alias(Cols.Run))
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
        )
        .collect()
        .pivot(Cols.Lib, index=(Cols.Run, Cols.Size))
        .lazy()
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
        .collect()
    )
