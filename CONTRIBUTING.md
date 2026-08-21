# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

### Python API and typing

- [pyochain/](pyochain/) — all stubs
- [pyochain/_types.pyi](pyochain/_types.pyi) — shared typing protocols and type aliases; it has no direct Rust module.
- [pyochain/pyochain.pyi](pyochain/pyochain.pyi) — top-level public re-exports for the extension initialized by [src/lib.rs](src/lib.rs).
- [pyochain/core/](pyochain/core/) — stubs for the core Rust types in [src/core/](src/core/).
- [pyochain/abc/](pyochain/abc/) — stubs for the ABCs and mixins in [src/abc/](src/abc/).
- [pyochain/collections/](pyochain/collections/) — stubs for the concrete collections in [src/collections/](src/collections/), including sorted collections.

The stub packages follow the public Rust module hierarchy, but the mapping is not strictly one-to-one: package initializers, grouped stubs, and private Rust helper modules do not always have a matching file.

### Rust and PyO3 implementation

- [src/lib.rs](src/lib.rs) — initializes the `pyochain` PyO3 module and registers the `core`, `abc`, `collections`, and `collections._sorted` submodules.
- [src/core/](src/core/) — implements the core types: `Dict`, `Iter`, `Peekable`, `Option`, `Result`, `Range`, `Seq`, `Set`, `SetMut`, `SliceView`, and `Vec`.
- [src/abc/](src/abc/) — implements the abstract base classes, mixins, and shared ABC traits.
- [src/collections/](src/collections/) — implements concrete collections such as `Deque`, `Heap`, `HeapMax`, `HeapMin`, `PyoCounter`, and `StableSet`.
- [src/collections/sorted/](src/collections/sorted/) — implements sorted collections, views, iterators, and their internal support modules.
- [src/display.rs](src/display.rs) — formats Python objects for `repr` output.
- [src/hasher.rs](src/hasher.rs) — provides shared hashing helpers.
- [src/traits.rs](src/traits.rs) — defines shared wrapper, conversion, and initialization traits.

### Internal crates

- [crates/pyo3_ext/](crates/pyo3_ext/) — internal PyO3 extensions and utility traits.
- [crates/pyochain_macros/](crates/pyochain_macros/) — procedural macros used by the Rust implementation.

### Tests, documentation, and tooling

- [tests/](tests/) — Python tests, ABC tests, external integration tests
- [benchmarks/](benchmarks/) — Python benchmarks for performance testing.
- [docs/](docs/) — documentation sources and API reference pages.
- [scripts/](scripts/) — documentation generation and repository validation scripts.
- [Cargo.toml](Cargo.toml) — Rust workspace and dependency configuration.
- [pyproject.toml](pyproject.toml) — Python package metadata, maturin configuration, and development dependencies.
- [pyrefly.toml](pyrefly.toml) — Pyrefly configuration.
- [ruff.toml](ruff.toml) — Ruff linting and formatting configuration.
- [zensical.toml](zensical.toml) — documentation site configuration.

## Stubs Docstrings

docstrings should follow the format below.

The code in the `examples` section will be automatically part of the test suite.

We use code blocks instead of doctests, so write them just like you would in a classic pytest file, i.e assertions.

```python
def my_function(param1: int, param2: str) -> bool:
    """One liner description of what the function does.

    Additional explanations if needed.

    List of points:
        - Point 1
        - Point 2
        - Point 3

    Args:
        param1 (int): Description.
        param2 (str): Description.

    Warning:
        Description of the warning.

    Note:
        Description of the note.

    Tip:
        Description of the tip.

    Returns:
        bool: Description of the return value.

    Examples:
        ```python
        assert my_function(5, "test")

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

For development mode (fast compile, no optimizations):

```bash
uv run maturin develop --uv
```

For benchmarking (optimized, slower compile):

```bash
uv run maturin develop --release --uv
```

Prior to a release, to check correct documentation generation, run the build tool:

```bash
.\target\release\pyochain-build.exe
```

If you are working on the build tool itself, you can build/run it directly with cargo:

```bash
cargo run -p pyochain-build
cargo run --release -p pyochain-build
cargo build -p pyochain-build
cargo build --release -p pyochain-build
```

To force a complete rebuild (clears all Rust artifacts):

```bash
cargo clean
uv run maturin develop --uv
```

## Tests and quality checks

Before committing, ensure all checks pass.

### type checking/linting/formatting

If `uv run -m scripts.check_docstrings` fails, don't worry.

`sdsort` will re-order the python stubs depending on various rules, so don't be surprised if your code moves around a bit.

```bash
uv run sdsort . --stubs;
uv run ruff check . --fix --unsafe-fixes;
uv run ruff format . --preview;
uv run basedpyright src/pyochain;
uv run scripts/check_docstrings.py;
uv run pydoclint pyochain/**/*.pyi
```

### tests

```bash
uv run pytest
```

### Building docs

To build and serve the documentation locally:

```shell
uv run zensical build -c
```

Then open your browser with the [site](site/index.html) to view the generated documentation.

### Benchmarks

Benchmarks are located in [tests/benchmarks/](tests/benchmarks) and use `pytest-benchmark`.
See [tests/benchmarks/README.md](tests/benchmarks/README.md) for details on running and interpreting benchmarks.

```shell
uv run pytest tests/benchmarks --benchmark-only --benchmark-warmup=True --benchmark-group-by=<name, param:<size>, group>
```

## Contributing workflow

- Create a branch per feature/fix and keep commits focused and descriptive.
- Run all quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
- For Rust changes, consider adding benchmarks to verify performance impact.

## Release process

### Changelogs and release template

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

### Issue on release

If an issue on a release appear, AND the package is NOT published on Pypi, running the following commands can help going back to a clean state without needing to create a new release:

```bash
git tag -d <tag_name>
git push origin --delete <tag_name>
```

This will convert the last tag into a draft release, allowing you to fix the issue and publish the release again without creating a new one.
