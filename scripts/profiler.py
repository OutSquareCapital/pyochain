"""Profile Python scripts with samply."""

from __future__ import annotations

import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def profiler(script: str, args: Iterable[str]) -> None:  # ruff: ignore[undocumented-public-function]
    path = Path(__file__).with_name("_boot.py")
    target = subprocess.Popen((sys.executable, path, script, *args))
    p_args = ("samply", "record", "-p", str(target.pid))
    samply = subprocess.Popen(p_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    _ = target.wait()
    samply.send_signal(signal.CTRL_BREAK_EVENT)
    with suppress(KeyboardInterrupt):
        _ = samply.wait()
    samply.kill()
