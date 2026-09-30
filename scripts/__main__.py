"""CLI entry point for Python scripts."""

from __future__ import annotations

from typing import Annotated

import typer as tp

app = tp.Typer(name="profile", help="Profile a Python script with samply.")


# BUG: need to kill the terminal manually after the script is done. CTRL + C won't work otherwise.
@app.command()
def profile(
    script: Annotated[str, tp.Argument(help="Python script to profile.")],
    args: Annotated[
        list[str] | None, tp.Argument(help="Arguments passed to the Python script.")
    ] = None,
) -> None:
    """Run a Python script and profile it with samply.

    Needed on Windows because it's not possible to attach samply directly to the running process.

    Without this, it's needed to manually handle two terminal at once.
    """
    from .profiler import profiler

    profiler(script, args or [])


if __name__ == "__main__":
    app()
