# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

Top-level files and folders of interest:

- `pyproject.toml` — Python project metadata and build configuration.
- `Cargo.toml` — Rust workspace and dependencies (root).
- `zensical.toml` — documentation site configuration.
- `README.md`, `CONTRIBUTING.md`, `LICENSE.md` — user-facing docs.
- `docs/` — project documentation sources (Markdown files).
- `site/` — generated documentation (static site).
- `scripts/` — development helpers (benchmarks, doc generation).
- `src/pyochain/` — Python package source code.
- `rust/` — Rust extension module (PyO3 bindings).
- `tests/` — Python test suite.

## Architecture: Hybrid Python/Rust

pyochain is a **mixed Python/Rust project**:

- **Python layer** (`src/pyochain/`) — Public API, pure Python implementations, type stubs.
- **Rust layer** (`rust/`) — Performance-critical code compiled to native bindings via PyO3.
- **PyO3 bridge** — Seamless integration between Python and Rust types.

### Rust structure (`rust/src/`)

- `lib.rs` — Crate root, module organization.
- `option.rs` — `Option[T]`, `Some`, `NONE` implementations (PyO3 classes).
- `result.rs` — `Result[T, E]`, `Ok`, `Err` implementations.
- `types.rs` — Utility functions and type conversions.

## Package layout

The `src/pyochain` Python package is organized into internal modules (leading underscore indicates internal implementation):

- `_iter.py` — `Iter`, `Seq`, `Vec`, `Set`, `SetMut` and iteration/collection methods.
- `_dict.py` — `Dict` class and mapping-related logic.
- `_range.py` — `Range` class impléàmentation.
- `rs.pyi` - Type stubs for public Rust-compiled bindings -> `Option`, `Result`, `Pipeable`, `Checkable`.
- `_tools.pyi` — Type stubs for iteration methods compiled to Rust. Internal, are called by public object methods, but not directly by users.
- `_types.py` — Protocols for element types (e.g., `SupportsLen`).
- `traits/` — abc's, e.g `PyoIterator`, `PyoMutableMapping`.
- `py.typed` — PEP 561 marker for type checking support.
- `__init__.py` — public API (imports and re-exports main classes).

## Coding and documentation guidelines

### Python code

- All public API functions and methods must include clear docstrings, full type hints, and overloads/generics where appropriate.
- Follow the existing docstring style: one-line summary, optional extended description, `Args:`, `Returns:`, `Examples:` sections. Keep a blank line between sections.
- Prefer simple, readable implementations. Hot paths use vanilla iteration to reduce call overhead.

### Rust code

- Use PyO3's modern Bound API (v0.27+).
- Add docstrings in `.pyi` stub files (auto-discovered by pytest doctests).

Docstring example:

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

After cloning the repo, set up the development environment (the project uses `uv` for both Python and Rust):

```bash
uv sync --dev
```

### Building the Rust extension

For development mode (debug symbols, faster compile):

```bash
uv run maturin develop --uv
```

For benchmarking (optimized, slower compile):

```bash
uv run maturin develop --release --uv
```

To force a complete rebuild (clears all Rust artifacts):

```bash
cd rust
cargo clean
cd ..
uv run maturin develop --release --uv
```

## Tests and quality checks

Before committing, ensure all checks pass:

```bash
uv run ruff check --fix src/pyochain;
uv run ruff format src/pyochain;
uv run basedpyright src/pyochain;
uv run scripts/check_docstrings.py;
uv run pydoclint src/pyochain;
uv run pytest --cov=src --cov-report=term-missing
```

## Benchmarks

Benchmarks are located in `tests/benchmarks/` and use `pytest-benchmark`. See `tests/benchmarks/README.md` for details on running and interpreting benchmarks.

```shell
uv run pytest tests/benchmarks --benchmark-only --benchmark-warmup=True --benchmark-group-by=<name, param:<size>, group>
```

## Building docs

To build and serve the documentation locally:

```bash
scripts\rebuild-docs.ps1
```

Then open your browser at the address shown.

## Contributing workflow

- Create a branch per feature/fix and keep commits focused and descriptive.
- Run all quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
- For Rust changes, consider adding benchmarks to verify performance impact.

## Issue on release

If an issue on a release appear, AND the package is NOT published on Pypi, running the following commands can help going back to a clean state without needing to create a new release:

```bash
git tag -d <tag_name>
git push origin --delete <tag_name>
```

This will convert the last tag into a draft release, allowing you to fix the issue and publish the release again without creating a new one.

## Changelogs and release template

Below is a template used for sections in CHANGELOG.md and release notes on GitHub.

When preparing a release, update the "unreleased" section with the relevant changes and then move it to a new section with the version number and release date.

```txt
# Pyochain v<VERSION>

## Changes

### 💥 Breaking changes

### 🏆 Highlights

### ⚠️ Deprecations

### 🚀 Performance improvements

### ✨ Enhancements

### 🐞 Bug fixes

### 📖 Documentation

### 🛠️ Other improvements

### 🔄 Refactors

### 📦 Build system

```
