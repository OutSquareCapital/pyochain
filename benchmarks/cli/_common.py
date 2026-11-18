from __future__ import annotations

import platform
import sys
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Final, NewType

import typer as tp
from rich.console import Console
from rich.text import Text

from pyochain import NONE, Dict, Err, Iter, Ok, Option, Result, Vec
from pyochain.core import Some

from .. import test_sorted

if TYPE_CHECKING:
    from pyochain.abc import PyoIterator

CONSOLE = Console()
PLATFORM_DIR: Final[str] = (
    f"{platform.system()}-CPython-{sys.version_info.major}.{sys.version_info.minor}-{platform.architecture()[0]}"
)
# NOTE: annoying repetition but wathever
RUN_PATH: Final[Path] = Path("benchmarks", "test_sorted")
SAVE_PATH: Final[Path] = Path(".benchmarks", "test_sorted")
"""Path to the benchmark results directory."""
GET_PATH: Final[Path] = SAVE_PATH.joinpath(PLATFORM_DIR)
"""Path to the benchmark results directory for the current platform (automatically appended by pytest-benchmark)."""
Method = NewType("Method", str)
"""A validated benchmark method name that corresponds to a test function."""


def check_method(method: str) -> Result[Method, tp.BadParameter]:
    available_methods = _get_test_funcs()
    if not available_methods.contains(method):
        methods = available_methods.iter().map(lambda m: " - " + m).join("\n")
        txt = Text("\nAvailable methods:\n", style="bold yellow").append(
            methods, style="bold green"
        )
        CONSOLE.print(txt)
        msg = f"Error: Method '{method}' not found in benchmark tests."
        return Err(tp.BadParameter(msg))
    else:
        return Ok(Method(method))


@cache
def _get_test_funcs() -> Vec[str]:
    return (
        Dict
        .from_object(test_sorted)
        .items()
        .iter()
        .filter_map_star(_add_module)
        .flatten()
        .sort()
    )


def _add_module(mod_name: str, obj: object) -> Option[PyoIterator[str]]:
    match obj:
        case ModuleType():
            return Some(
                Iter(obj.__dict__)
                .filter(lambda fn: fn.startswith("test_"))
                .map(lambda fn: f"{mod_name}.py::{fn}")
            )
        case _:
            return NONE
