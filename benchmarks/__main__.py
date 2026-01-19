"""Entry point for benchmarks CLI."""

from typing import Annotated

import typer

from . import benchs  # pyright: ignore[reportUnusedImport] # noqa: F401
from ._pipeline import Benchmarks, run_pipeline
from ._registery import CONSOLE

app = typer.Typer(help="Benchmarks for pyochain developments.")


@app.command()
@Benchmarks.db
def main(
    *,
    setup: Annotated[
        bool, typer.Option("--setup", help="Setup database on first run")
    ] = False,
) -> None:
    """Run benchmarks and persist results to database."""
    if setup:
        Benchmarks.source().mkdir(exist_ok=True)
    CONSOLE.print("Running benchmarks...", style="bold blue")
    df = run_pipeline()

    if setup:
        Benchmarks.db.results.create_or_replace_from(df)
    else:
        Benchmarks.db.results.insert_into(df)
    CONSOLE.print()
    CONSOLE.print("✓ Results persisted to database", style="bold green")


if __name__ == "__main__":
    app()
