# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

Top-level files and folders of interest:

- [pyproject.toml](pyproject.toml) — Python project metadata and build configuration.
- [rust/Cargo.toml](rust/Cargo.toml) — Rust crate metadata and dependencies.
- [zensical.toml](zensical.toml) — documentation site configuration.
- [README.md](README.md), [CONTRIBUTING.md](CONTRIBUTING.md), [LICENSE.md](LICENSE.md) — user-facing docs.
- [docs/](docs) — project documentation sources.
- [site/](site) — generated documentation output.
- [scripts/](scripts) — development helpers for docs and checks.
- [src/pyochain/](src/pyochain) — Python package source code.
- [rust/](rust) — Rust extension module built with PyO3.
- [tests/](tests) — Python test suite and benchmarks.

## Architecture: Hybrid Python/Rust

pyochain is a **mixed Python/Rust project**:

- **Public Python package**: [src/pyochain/](src/pyochain) contains the import surface, collection types, ABCs, and type stubs.
- **Native Rust extension**: [rust/src/](rust/src) contains the PyO3 module compiled as `pyochain.rs` via [pyproject.toml](pyproject.toml) and [rust/Cargo.toml](rust/Cargo.toml).

### Python structure ([src/pyochain/](src/pyochain))

- [src/pyochain/**init**.py](src/pyochain/__init__.py) — public API entrypoint and re-exports.
- [src/pyochain/_iter.py](src/pyochain/_iter.py) — `Iter`, `Seq`, `Vec`, `Set`, `SetMut`, `Peekable`, and related collection logic.
- [src/pyochain/_dict.py](src/pyochain/_dict.py) — `Dict` implementation and mapping-specific methods.
- [src/pyochain/_range.py](src/pyochain/_range.py) — `Range` implementation.
- [src/pyochain/traits/](src/pyochain/traits) — abstract collection and iterator ABCs shared across the Python layer.
- [src/pyochain/rs.pyi](src/pyochain/rs.pyi) — stubs for the Rust-compiled public bindings.
- [src/pyochain/_tools.pyi](src/pyochain/_tools.pyi) — stubs for the internal Rust helper module exposed as `pyochain._tools`.
- [src/pyochain/_types.py](src/pyochain/_types.py) — shared typing protocols and support types.

### Rust structure ([rust/src/](rust/src))

- [rust/src/lib.rs](rust/src/lib.rs) — PyO3 module root; registers the exported classes, functions, and the `_tools` submodule.
- [rust/src/option.rs](rust/src/option.rs) — `Option[T]`, `Some`, `Null`, `NONE`, and helper constructors.
- [rust/src/result.rs](rust/src/result.rs) — `Result[T, E]`, `Ok`, and `Err` implementations.
- [rust/src/errors.rs](rust/src/errors.rs) — unwrap error types exposed to Python.
- [rust/src/mixins.rs](rust/src/mixins.rs) — mixins types `Checkable` and `Pipeable`.
- [rust/src/tools.rs](rust/src/tools.rs) — performance-oriented iterator helpers exposed through `pyochain._tools`.
- [rust/src/args.rs](rust/src/args.rs) and [rust/src/hasher.rs](rust/src/hasher.rs) — argument parsing and hashing utilities used by the extension.

### Build and integration

- [pyproject.toml](pyproject.toml) configures `maturin` as the build backend and maps the compiled module name to `pyochain.rs`.
- [rust/Cargo.toml](rust/Cargo.toml) defines the Rust crate compiled as a `cdylib`.
- [rust/src/lib.rs](rust/src/lib.rs) injects the Rust `_tools` module into Python's import system so the Python layer can use it as `pyochain._tools`.

## Package layout

## Coding and documentation guidelines

### Python code

- All public API functions and methods must include clear docstrings, full type hints, and overloads/generics where appropriate.
- Follow the existing docstring style: one-line summary, optional extended description, `Args:`, `Returns:`, `Examples:` sections. Keep a blank line between sections.
- Prefer simple, readable implementations. Hot paths use vanilla iteration to reduce call overhead.

### Rust code

- Use PyO3's modern Bound API (v0.27+).
- Add docstrings in [src/pyochain/rs.pyi](src/pyochain/rs.pyi) for all public Rust types and methods. The `_tools` module is already tested by the caller site (e.g public classes methods)

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

Before committing, ensure all checks pass.

### type checking/linting/formatting

```bash
uv run ruff check --fix src/pyochain;
uv run ruff format src/pyochain;
uv run basedpyright src/pyochain;
```

Unfortunately, `Ruff` doesn't work well when doctests are mixed with backticks sections in docstrings to format code examples.

The workaround is to temporarily remove them, run `Ruff` and then put them back.

For multiple sections, you can use your IDE to replace both of them by dummy text, run `Ruff` and then replace the dummy text by the original backticks.

### tests

Add `--cov=src --cov-report=term-missing` to the pytest command below to include coverage reports.

```bash
uv run scripts/check_docstrings.py;
uv run pydoclint src/pyochain;
uv run pytest
```

## Benchmarks

Benchmarks are located in [tests/benchmarks/](tests/benchmarks) and use `pytest-benchmark`. See [tests/benchmarks/README.md](tests/benchmarks/README.md) for details on running and interpreting benchmarks.

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

### 🆕 New features

### 🚀 Performance improvements

### ⚠️ Performance regressions

### ✨ Enhancements

### 🐞 Bug fixes

### 📖 Documentation

### 🛠️ Other improvements

### 🔄 Refactors

### 📦 Build system

### 🔗 Dependencies

### 🧪 Tests
```
