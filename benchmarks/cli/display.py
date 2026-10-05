import plotly.express as px
import polars as pl
from rich.markdown import Markdown

from ._common import CONSOLE, Lib
from .query import Cols, group_by_commit

cfg = pl.Config(
    tbl_formatting="MARKDOWN",
    tbl_hide_column_data_types=True,
    fmt_str_lengths=200,
    tbl_hide_dataframe_shape=True,
    set_tbl_rows=1000,
)


@cfg
def terminal(df: pl.DataFrame, *, raw: bool) -> None:
    _show(df, "All results details", raw=raw)
    _ = (
        df
        .lazy()
        .pipe(group_by_commit)
        .drop(Cols.Run, Cols.TimeStamp)
        .collect()
        .pipe(_show, "Aggregated by commit", raw=raw)
    )


def _show(df: pl.DataFrame, title: str, *, raw: bool) -> None:
    CONSOLE.rule(title)
    df_repr = str(df)
    if raw:
        return CONSOLE.print(df_repr)
    else:
        return CONSOLE.print(Markdown(df_repr))


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
