"""Run code checks."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from rich.console import Console
from rich.text import Text

from pyochain import Err, Null, Ok, Option, Result, Seq, Some, Vec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyochain.abc import PyoSequence

CACHE: Final[Path] = Path(".cache", "pyochain-check")
CONSOLE: Final[Console] = Console()


@dataclass(slots=True)
class Command:
    """A command to run: base args, plus the flags specific to check mode and fix mode."""

    base: str
    check_flags: str = ""
    fix_flags: str = ""

    def command(self, *, fix: bool) -> Seq[str]:  # ruff: ignore[undocumented-public-method]
        flags = self.fix_flags if fix else self.check_flags
        return Seq(*self.base.split(), *flags.split())


def run(*, fix: bool, slow: bool) -> int:
    """Run all code checks in order, stopping at the first failure and caching the index of the failed check.

    Returns:
        int: 0 if all checks pass, 1 if a check fails.
    """
    start = int(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else 0
    return (
        _commands(slow=slow)
        .iter()
        .map(lambda cmd: cmd.command(fix=fix))
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


def _commands(*, slow: bool) -> PyoSequence[Command]:
    commands = _fast_commands()
    if slow:
        commands.extend(_slow_commands())
    return commands


def _fast_commands() -> Vec[Command]:
    return Vec(
        Command("cargo fmt --all", check_flags="-- --check"),
        Command("uv run sdsort . --stubs", check_flags="--check"),
        Command("uv run ruff check .", fix_flags="--fix --unsafe-fixes"),
        Command("uv run ruff format . --preview", check_flags="--check"),
        Command("uv run pydoclint pyochain/**/*.pyi"),
        Command("uv run tombi format", check_flags="--check"),
        Command("uv run tombi lint"),
    )


def _slow_commands() -> Sequence[Command]:
    return (
        Command("cargo clippy --workspace", fix_flags="--fix --allow-dirty"),
        Command("uv run basedpyright ."),
    )
