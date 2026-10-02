"""Run code checks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from enum import StrEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Final, Self

from rich.console import Console
from rich.text import Text

from pyochain import Err, Iter, Null, Ok, Option, Result, Some, Vec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyochain.abc import PyoSequence

CACHE: Final[Path] = Path(".cache", "pyochain-check")
CONSOLE: Final[Console] = Console()
STUBS_PATH: Final[Path] = Path("pyochain")


def run(*, fix: bool, slow: bool) -> int:
    """Run all code checks in order, stopping at the first failure and caching the index of the failed check.

    Returns:
        int: 0 if all checks pass, 1 if a check fails.
    """
    start = int(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else 0
    return (
        _all_commands(slow=slow)
        .iter()
        .map(lambda cmd: cmd.build(fix=fix))
        .enumerate()
        .skip(start)
        .find_map(_run_tool)
        .transpose()
        .pipe(_handle_result)
    )


def _handle_result(result: Result[Option[None], str]) -> int:
    match result:
        case Err(index):
            CACHE.parent.mkdir(exist_ok=True)
            _ = CACHE.write_text(index, encoding="utf-8")
            return 1
        case Ok(_):
            CACHE.unlink(missing_ok=True)
            return 0


def _run_tool(args: tuple[int, PyoSequence[str]]) -> Option[Result[None, str]]:
    index, command = args
    msg = command.iter().join(" ")
    CONSOLE.rule(Text(msg, style="bold green"), style="bold yellow")
    match subprocess.run(command, check=False).returncode:
        case 0:
            return Null()
        case _:
            return Some(Err(str(index)))


def _all_commands(*, slow: bool) -> PyoSequence[_CommandBuilder]:
    commands = _fast_commands()
    if slow:
        commands.extend(_slow_commands())
    return commands


def _fast_commands() -> Vec[_CommandBuilder]:
    # NOTE: pydoclint next release will handle stubs files natively.
    stubs_files = Iter(STUBS_PATH.rglob("*.pyi")).map(str)
    return Vec(
        _Tools.FMT.do("--all").check_args("--", "--check"),
        _Tools.SDSORT.do(".", "--stubs").check_args("--check"),
        _Tools.RUFF.do("check", ".").fix_args("--fix", "--unsafe-fixes"),
        _Tools.RUFF.do("format", ".", "--preview").check_args("--check"),
        _Tools.PYDOCLINT.do(*stubs_files),
        _Tools.TOMBI.do("format").check_args("--check"),
        _Tools.TOMBI.do("lint"),
    )


def _slow_commands() -> Sequence[_CommandBuilder]:
    return (
        _Tools.CLIPPY.do("--workspace").fix_args("--fix", "--allow-dirty"),
        _Tools.BASEDPYRIGHT.do("."),
    )


@dataclass(slots=True)
class _CommandBuilder:
    """A command to run."""

    args: Vec[str]
    """Arguments always passed."""
    _check_args: Vec[str] = field(default_factory=Vec[str])
    """Arguments passed when running in check mode."""
    _fix_args: Vec[str] = field(default_factory=Vec[str])
    """Arguments passed when running in fix mode."""

    def check_args(self, *args: str) -> Self:
        self._check_args.extend(args)
        return self

    def fix_args(self, *args: str) -> Self:
        self._fix_args.extend(args)
        return self

    def build(self, *, fix: bool) -> PyoSequence[str]:
        self.args.extend(self._fix_args if fix else self._check_args)
        return self.args


class _Tools(StrEnum):
    """Tools used for code checks."""

    RUFF = auto()
    PYDOCLINT = auto()
    TOMBI = auto()
    SDSORT = auto()
    BASEDPYRIGHT = auto()
    FMT = auto()
    CLIPPY = auto()

    def do(self, *args: str) -> _CommandBuilder:
        match self:
            case self.FMT | self.CLIPPY:
                return _CommandBuilder(Vec("cargo", self, *args))
            case py_tool:
                return _CommandBuilder(Vec("uv", "run", py_tool, *args))
