from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from pyochain import Err, Ok, Result

from ._common import RUN_PATH, SAVE_PATH, Method

if TYPE_CHECKING:
    from collections.abc import Sequence


def run(method: Method, repeat: int) -> Result[None, ValueError]:
    match repeat:
        case _ if repeat < 1:
            txt = f"Repeat must be a positive integer, got {repeat}."
            return Err(ValueError(txt))
        case 1:
            _ = subprocess.run(_args(method), check=True)
            return Ok(None)
        case _:
            for _ in range(repeat):
                _ = subprocess.run(_args(method), check=True)
            return Ok(None)


def _args(method: Method) -> Sequence[str]:
    return (
        "pytest",
        f"{RUN_PATH.as_posix()}::test_{method}",
        _bench_arg("only"),
        _bench_arg("warmup=true"),
        _bench_arg("disable-gc"),
        _bench_arg("autosave"),
        _bench_arg(f"storage=file://{SAVE_PATH.as_posix()}"),
    )


def _bench_arg(arg: str) -> str:
    return f"--benchmark-{arg}"
