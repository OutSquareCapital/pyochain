"""CLI entry point for Python scripts."""

from __future__ import annotations

from typing import Annotated

import typer as tp

app = tp.Typer(name="profile", help="General dev tools.")


# BUG: need to kill the terminal manually after the script is done. CTRL + C won't work otherwise.
@app.command()
def profile() -> None:
    """Profile `t.py` with `samply`, for performance analysis of Rust calls from Python.

    Needed on Windows because it's not possible to attach samply directly to the running process.

    Make sure to set up `time.sleep(x)` with a few seconds at the top of the script.

    It gives you the time to accept the pop-up asking to give the permission to `samply` to modify things.

    Without this, it's needed to manually handle two terminal at once.
    """
    from .profiler import profiler

    profiler()


@app.command()
def ci(
    *,
    fix: Annotated[
        bool, tp.Option(help="Apply autofixes instead of check-only")
    ] = False,
    slow: Annotated[
        bool,
        tp.Option(
            help="""Run check flagged as slow (several seconds to run).\n
            Changing this between two runs will reset the cache index."""
        ),
    ] = False,
) -> None:
    """Run CI checks, with caching of the last failed step.

    Switching from `--slow` to `--no-slow` (or vice versa) will reset the cache index.

    Note that the caching is strictly for local development convenience. Adding new commands in it will invalidate the cache index.

    Raises:
        Exit: If a check fails or all checks pass, with the corresponding exit code.
    """
    from . import checks

    raise tp.Exit(checks.run(fix=fix, slow=slow))


if __name__ == "__main__":
    app()
