from __future__ import annotations

import platform
import sys
from enum import StrEnum, auto
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, NewType, TypeIs

import typer as tp
from rich.console import Console
from rich.text import Text

from pyochain import Dict, Err, Iter, Ok, Result, Vec

from .. import test_sorted

if TYPE_CHECKING:
    import polars as pl

    from pyochain.abc import PyoIterator

CONSOLE = Console()
PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
TEST = "test_"
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
        Dict
        .from_object(test_sorted)
        .items()
        .iter()
        .filter_star(lambda _, v: _is_module(v))
        .map_star(_add_module)
        .flatten()
        .sort()
    )


def _add_module(mod_name: str, module: ModuleType) -> PyoIterator[str]:
    name = f"{mod_name.removeprefix(TEST)}::"
    return (
        Iter(module.__dict__)
        .filter(lambda fn: fn.startswith(TEST))
        .map(lambda fn: f"{name}{fn.removeprefix(TEST)}")
    )


def _is_module(v: object) -> TypeIs[ModuleType]:
    return isinstance(v, ModuleType)
