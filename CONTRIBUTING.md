# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

Top-level files and folders of interest:

- `pyproject.toml` — project metadata and build configuration.
- `zensical.toml` — documentation site configuration.
- `README.md`, `CONTRIBUTING.md`, `LICENSE.md`, — user-facing docs.
- `docs/` — project documentation sources.
- `scripts/` — development helpers and benchmarks.
- `src/pyochain/` — package source (public API defined in `src/pyochain/__init__.py`).

## Package layout

The `src/pyochain` package is organized into a small number of internal modules (leading underscore indicates internal implementation):

- `_iter.py` — implementation of `Iter` (lazy iterator), `Seq`, `Vec`, `Set`, `SetMut` and all iteration/collection helpers.
- `_dict.py` — `Dict` class implementation and mapping-related logic.
- `_option.py` — `Option`, `Some`, `NONE`, and `OptionUnwrapError`.
- `_result.py` — `Result`, `Ok`, `Err`, and `ResultUnwrapError`.
- `_types.py` — Protocols for elements types returned by various methods (e.g., `SupportsLen`).
- `traits/` — public mixin traits for fluent chaining (`Pipeable`, `Checkable`) that can be added to custom types.
- `py.typed` — PEP 561 marker file for type checking support.

The public API is exposed through `src/pyochain/__init__.py`, which imports and re-exports the main classes, types, and traits.

## Coding and documentation guidelines

- All public API functions and methods must include clear docstrings, full type hints, and overloads/generics where appropriate.
- Follow the existing docstring style used in the repository: one-line summary, optional extended description, `Args:`, `Returns:`, and `Examples:` sections. Keep a blank line between sections.
- Prefer simple, readable implementations in internal modules. The project favors "vanilla" Python iteration in hot paths to reduce call overhead.

Docstring example (follow the repository's style):

```python
def my_function(param1: int, param2: str) -> bool:
        """One liner description of what the function does.

        Args:
                param1 (int): Description.
                param2 (str): Description.

        Returns:
                bool: Description of the return value.

        Example:
        ```python
        >>> my_function(5, "test")
        True
        ```
        """
        return True
```

## Setup

After cloning the repo, set up the development environment (the project uses `uv` tasks in this repo):

```bash
uv sync --dev
```

## Tests and quality checks

Before committing, ensure tests and quality checks pass.

```bash
uv run ruff check --fix src/pyochain
uv run ruff format src/pyochain
uv run pydoclint src/pyochain
uv run pyright src/pyochain
uv run pytest --doctest-modules src/pyochain
uv run pytest tests/
uv run pytest README.md --doctest-glob="*.md" --doctest-mdcodeblocks -v
uv run pytest docs/ --doctest-glob="*.md" --doctest-mdcodeblocks -v
```

## Building docs

To build the documentation locally, run:

```bash
scripts\rebuild-docs.ps1
```

Then open your browser at the given adress.

## Contributing workflow

- Create a branch per change and keep commits focused and descriptive.
- Run the quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
