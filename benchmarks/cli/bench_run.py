from __future__ import annotations

import subprocess

from pyochain import Err, Ok, Result

from ._common import CONSOLE, RUN_PATH, SAVE_PATH, Method


def _bench_arg(arg: str) -> str:
    return f"--benchmark-{arg}"


ARGS = (
    _bench_arg("only"),
    _bench_arg("warmup=true"),
    _bench_arg("disable-gc"),
    _bench_arg("autosave"),
    _bench_arg("quiet"),
    _bench_arg(f"storage=file://{SAVE_PATH.as_posix()}"),
)


def run(method: Method, repeat: int, k: str) -> Result[None, ValueError]:
    match repeat:
        case _ if repeat < 1:
            txt = f"Repeat must be a positive integer, got {repeat}."
            return Err(ValueError(txt))
        case 1:
            return Ok(_inner_run(method, k))
        case _:
            for i in range(repeat):
                CONSOLE.rule(f"Run {i + 1}/{repeat}", style="bold yellow")
                _ = _inner_run(method, k)
            return Ok(None)


def _inner_run(method: Method, k: str) -> None:
    args = ["pytest", RUN_PATH.joinpath(method).as_posix()]
    args.extend(ARGS)
    if len(k) > 0:
        args.extend(("-k", k))
    _ = subprocess.run(args, check=True)
