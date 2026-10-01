from __future__ import annotations

import platform
import sys
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Final, NewType

import typer as tp
from rich.console import Console
from rich.text import Text

from pyochain import Err, Iter, Ok, Result, Vec

from .. import test_sorted

if TYPE_CHECKING:
    import polars as pl

CONSOLE = Console()
PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
PREFIX = "test_"
"""Test function prefix used to identify benchmark tests."""
# NOTE: annoying repetition but wathever
RUN_PATH: Final[Path] = Path("benchmarks", "test_sorted.py")
SAVE_PATH: Final[Path] = Path(".benchmarks", "test_sorted")
"""Path to the benchmark results directory."""
GET_PATH: Final[Path] = SAVE_PATH.joinpath(PLATFORM_DIR)
"""Path to the benchmark results directory for the current platform (automatically appended by pytest-benchmark)."""
Method = NewType("Method", str)
"""A validated benchmark method name that corresponds to a test function."""


class PlEnum(StrEnum):
    """Base class for enums that can be used as polars expressions."""

    def pl(self) -> pl.Expr:
        """Return a polars expression for the enum value."""

        import polars as pl

        return pl.col(self.value)


class Lib(PlEnum):
    """Libraries used in the benchmarks."""

    Pyochain = auto()
    SortedContainers = auto()


def check_method(method: str) -> Result[Method, tp.BadParameter]:
    available_methods = _get_test_funcs()
    if not available_methods.contains(method):
        methods = available_methods.iter().map(str).map(lambda m: " - " + m).join("\n")
        txt = Text("\nAvailable methods:\n", style="bold yellow").append(
            methods, style="bold green"
        )
        CONSOLE.print(txt)
        msg = f"Error: Method '{method}' not found in benchmark tests."
        return Err(tp.BadParameter(msg))
    else:
        return Ok(Method(method))


def _get_test_funcs() -> Vec[str]:
    return (
        Iter(test_sorted.__dict__)
        .filter(lambda name: name.startswith(PREFIX))
        .map(lambda name: name.removeprefix(PREFIX))
        .sort()
    )
