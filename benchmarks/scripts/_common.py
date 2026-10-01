from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Final, NewType

from rich.console import Console
from rich.text import Text

from pyochain import Err, Iter, Ok, Result, Vec
from tests.test_extern import test_sorted

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


def check_method(method: str) -> Result[Method, ValueError]:
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
        return Err(ValueError(""))
    else:
        return Ok(Method(method))


def _get_test_funcs() -> Vec[str]:
    return (
        Iter(test_sorted.__dict__)
        .filter(lambda name: name.startswith(PREFIX))
        .map(lambda name: name.removeprefix(PREFIX))
        .sort()
    )
