from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from pyochain import Err, Ok, Result

from ._common import CONSOLE, RUN_PATH, SAVE_PATH, Lib, Method

if TYPE_CHECKING:
    from pyochain.abc import PyoIterable


UV_RUN = ("uv", "run")


def run_many(commits: PyoIterable[str], method: Method, repeat: int) -> None:
    return (
        commits
        .iter()
        .map(_setup_and_test_commit)
        .try_for_each(lambda _: run(method, repeat, calibrate=False))
        .map(
            lambda _: CONSOLE.print(
                f"Benchmarks for {method} completed successfully.", style="bold green"
            )
        )
        .unwrap()
    )


def _setup_and_test_commit(commit: str) -> None:
    _ = subprocess.run(("git", "switch", commit), check=True)
    _ = subprocess.run((*UV_RUN, "maturin", "develop", "--release"), check=True)
    _ = subprocess.run((*UV_RUN, "pytest"), check=True)


def run(method: Method, repeat: int, *, calibrate: bool) -> Result[None, ValueError]:
    match repeat:
        case _ if repeat < 1:
            txt = f"Repeat must be a positive integer, got {repeat}."
            return Err(ValueError(txt))
        case 1:
            return Ok(_inner_run(method, calibrate=calibrate))
        case _:
            for _ in range(repeat):
                _ = _inner_run(method, calibrate=calibrate)
            return Ok(None)


def _inner_run(method: Method, *, calibrate: bool) -> None:
    args = [
        "pytest",
        f"{RUN_PATH.as_posix()}::test_{method}",
        _bench_arg("only"),
        _bench_arg("warmup=true"),
        _bench_arg("disable-gc"),
        _bench_arg("autosave"),
        _bench_arg(f"storage=file://{SAVE_PATH.as_posix()}"),
    ]
    if calibrate:
        args.extend(("-k", Lib.Pyochain))
    _ = subprocess.run(args, check=True)


def _bench_arg(arg: str) -> str:
    return f"--benchmark-{arg}"
