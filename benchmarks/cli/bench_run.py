from __future__ import annotations

import subprocess

from pyochain import Err, Ok, Result

from .._utils import Lib
from ._common import CONSOLE, RUN_PATH, SAVE_PATH, Method


def run(method: Method, repeat: int, *, calibrate: bool) -> Result[None, ValueError]:
    match repeat:
        case _ if repeat < 1:
            txt = f"Repeat must be a positive integer, got {repeat}."
            return Err(ValueError(txt))
        case 1:
            return Ok(_inner_run(method, calibrate=calibrate))
        case _:
            for i in range(repeat):
                CONSOLE.rule(f"Run {i + 1}/{repeat}", style="bold yellow")
                _ = _inner_run(method, calibrate=calibrate)
            return Ok(None)


def _inner_run(method: Method, *, calibrate: bool) -> None:
    args = [
        "pytest",
        RUN_PATH.joinpath(method).as_posix(),
        _bench_arg("only"),
        _bench_arg("warmup=true"),
        _bench_arg("disable-gc"),
        _bench_arg("autosave"),
        _bench_arg("quiet"),
        _bench_arg(f"storage=file://{SAVE_PATH.as_posix()}"),
    ]
    if not calibrate:
        args.extend(("-k", Lib.Pyochain))
    _ = subprocess.run(args, check=True)


def _bench_arg(arg: str) -> str:
    return f"--benchmark-{arg}"
