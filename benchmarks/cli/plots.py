import plotly.express as px
import polars as pl

from ._common import Lib
from .query import Cols


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
