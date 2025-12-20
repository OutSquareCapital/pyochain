# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

Top-level files and folders of interest:

- `pyproject.toml` — project metadata and build configuration.
- `mkdocs.yml` — documentation site configuration.
- `README.md`, `CONTRIBUTING.md`, `LICENSE.md`, `EXAMPLES.md` — user-facing docs.
- `docs/` — project documentation sources.
- `scripts/` — development helpers and benchmarks.
- `src/pyochain/` — package source (public API defined in `src/pyochain/__init__.py`).

## Package layout

The `src/pyochain` package is organized into a small number of internal subpackages (leading underscore indicates internal implementation):

- `_core/`:
  - `__init__.py`
  - `_config.py`
  - `_main.py` — common base classes such as `CommonBase`, `IterWrapper`, `MappingWrapper` and shared helpers.
  - `_protocols.py` — structural `typing.Protocol` definitions used across the package.

- `_iter/`:
  - `__init__.py`
  - `_common.py` — shared helpers and common methods used by both lazy and eager implementations.
  - `_lazy.py` — implementation of `Iter` (lazy iterator) and lazy iteration helpers.
  - `_eager.py` — implementation of `Seq`, `Vec` and eager collection helpers.

- `_dict/`:
  - `__init__.py`
  - `_filters.py`
  - `_groups.py`
  - `_iter.py`
  - `_joins.py`
  - `_main.py` — `Dict` class and constructor logic.
  - `_maps.py`
  - `_nested.py`
  - `_process.py`

- `_results/`:
  - `__init__.py`
  - `_option.py` — `Option`, `Some`, `NONE`, and `OptionUnwrapError`.
  - `_result.py` — `Result`, `Ok`, `Err`, and `ResultUnwrapError`.

Each of the method files (for example `_filters.py`, `_maps.py`) typically declares a small base class that groups related methods; the public class composes those bases via multiple inheritance in the corresponding `_main.py`.

**Note**: The above structure is subject to change as the project evolves and may be not up-to-date with the latest changes.

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

        Examples:
        ```python
        >>> my_function(5, "test")
        True
        ```
        """
        return True
```

## Setup

After cloning the repo, set up the development environment (the project uses `uv` tasks in this repo):

```powershell
uv sync --dev
```

## Tests and quality checks

Before committing, ensure tests and quality checks pass.

```powershell
uv run pydoclint src/pyochain
uv run ruff check --fix src/pyochain
uv run ruff format src/pyochain
uv run pyright src/pyochain
uv run pytest --doctest-modules src/pyochain
```

Notes:

- `pydoclint` checks docstrings.
- `ruff` is used for linting and formatting.
- `pyright` provides static type checking.
- `pytest --doctest-modules` runs doctests embedded in docstrings.

## Contributing workflow

- Create a branch per change and keep commits focused and descriptive.
- Run the quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
