import plotly.express as px
import polars as pl

from ._common import CONSOLE, Lib
from .query import Cols


@pl.Config(
    tbl_hide_column_data_types=True,
    fmt_str_lengths=200,
    tbl_hide_dataframe_shape=True,
    set_tbl_rows=1000,
)
def terminal(df: pl.DataFrame) -> None:
    CONSOLE.rule("All results details")
    CONSOLE.print(df)
    CONSOLE.rule("Aggregated results")
    _ = (
        df
        .lazy()
        .drop(Cols.TimeStamp)
        .group_by(Cols.Size, maintain_order=True)
        .agg(pl.selectors.numeric().median())
        .collect()
        .pipe(CONSOLE.print)
    )


def absolute(df: pl.DataFrame, method: str, x_axis: Cols) -> None:
    return px.line(  # pyright: ignore[reportUnknownMemberType]
        df,
        title=f"{method}: absolute speed across runs",
        x=x_axis,
        y=Lib.Pyochain,
        color=Cols.Size,
        log_y=True,
        template="plotly_dark",
    ).show()


def relative(df: pl.DataFrame, method: str, x_axis: Cols) -> None:
    return (
        px
        .line(  # pyright: ignore[reportUnknownMemberType]
            df,
            title=f"{method}: speedup of pyochain vs sortedcontainers across runs",
            x=x_axis,
            y=Cols.Relative,
            color=Cols.Size,
            template="plotly_dark",
        )
        .add_hline(y=1)
        .show()
    )
