"""Entry point for benchmarks CLI."""

from typing import Annotated

import typer

from . import benchs  # pyright: ignore[reportUnusedImport] # noqa: F401
from ._pipeline import Benchmarks, run_pipeline
from ._registery import CONSOLE

app = typer.Typer(help="Benchmarks for pyochain developments.")


@app.command()
@Benchmarks.db
def show() -> None:
    """Show all benchmark results from the database."""
    return (
        Benchmarks.db.results.scan()
        .select("category", "name", "size", "runs", "median", "git_hash")
        .pipe(print)
    )


@app.command()
@Benchmarks.db
def setup(
    *,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing benchmark database.")
    ] = False,
) -> None:
    """Setup the benchmark database."""
    Benchmarks.source().mkdir(exist_ok=True)
    if overwrite:
        Benchmarks.db.results.create_or_replace()
    else:
        Benchmarks.db.results.create()
    CONSOLE.print("✓ Benchmark database setup complete", style="bold green")


@app.command()
@Benchmarks.db
def run(
    *,
    debug: Annotated[
        bool, typer.Option("--dry", help="Don't persist results to database.")
    ] = False,
) -> None:
    """Run benchmarks and persist results to database."""
    CONSOLE.print("Running benchmarks...", style="bold blue")
    match debug:
        case True:
            CONSOLE.print("✓ Debug mode: results not persisted", style="bold yellow")
            return run_pipeline().pipe(print)
        case False:
            run_pipeline().pipe(Benchmarks.db.results.insert_into)

            CONSOLE.print()
            return CONSOLE.print("✓ Results persisted to database", style="bold green")


if __name__ == "__main__":
    app()
