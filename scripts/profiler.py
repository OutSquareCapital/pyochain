"""Profile Python scripts with samply."""

from __future__ import annotations

import signal
import subprocess
import sys
from contextlib import suppress


def profiler() -> None:  # ruff: ignore[undocumented-public-function]
    target = subprocess.Popen((sys.executable, "t.py"))
    p_args = ("samply", "record", "-p", str(target.pid))
    samply = subprocess.Popen(p_args, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    _ = target.wait()
    samply.send_signal(signal.CTRL_BREAK_EVENT)
    with suppress(KeyboardInterrupt):
        _ = samply.wait()
    samply.kill()
