# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.
nchmarks.

## Repository overview

pyochain is a **mixed Python/Rust project**:

### Python structure ([src/pyochain/](src/pyochain))

- [src/pyochain/**init**.py](src/pyochain/__init__.py) — public API entrypoint and re-exports.
- [src/pyochain/abc/](src/pyochain/abc) — abstract collections and iterator ABCs shared across the Python layer.
- [src/pyochain/collections/](src/pyochain/collections) — non-core collection implementations.
- [src/pyochain/_iter.py](src/pyochain/_iter.py) — `Iter` implementation.
- [src/pyochain/_seq.py](src/pyochain/_seq.py) — `Seq` implementation.
- [src/pyochain/_dict.py](src/pyochain/_dict.py) — `Dict` implementation and mapping-specific methods.
- [src/pyochain/_range.py](src/pyochain/_range.py) — `Range` implementation.
- [src/pyochain/_set.py](src/pyochain/_set.py) — `Set`, `SetMut`, `PyoKeysView`, `PyoValuesView`, and `PyoItemsView` implementations.
- [src/pyochain/_vec.py](src/pyochain/_vec.py) — `Vec` implementation.
- [src/pyochain/_utils.py](src/pyochain/_utils.py) — internal utilities used across the Python package.
- [src/pyochain/rs.pyi](src/pyochain/rs.pyi) — stubs for the Rust-compiled public bindings.
- [src/pyochain/_tools.pyi](src/pyochain/_tools.pyi) — stubs for the internal Rust helper module exposed as `pyochain._tools`.
- [src/pyochain/_types.py](src/pyochain/_types.py) — shared typing protocols and support types.

### Rust structure ([rust/src/](rust/src))

- [rust/src/lib.rs](rust/src/lib.rs) — PyO3 module root; exposes the `Option`/`Result` family, mixins, helper functions, the `NONE` constant, and the `_tools` submodule.
- [rust/src/option.rs](rust/src/option.rs) — `Option[T]`, `Some`, `Null`, `NONE`, and helper constructors.
- [rust/src/result.rs](rust/src/result.rs) — `Result[T, E]`, `Ok`, and `Err` implementations.
- [rust/src/errors.rs](rust/src/errors.rs) — unwrap error types exposed to Python.
- [rust/src/mixins.rs](rust/src/mixins.rs) — mixin types `Checkable` and `Pipeable`.
- [rust/src/tools.rs](rust/src/tools.rs) — iteration function and structs exposed through `pyochain._tools`.
- [rust/src/args.rs](rust/src/args.rs) and [rust/src/hasher.rs](rust/src/hasher.rs) — internal argument parsing and hashing utilities used by the extension.

## Coding and documentation guidelines

### Python code

#### Public API

All public API functions and methods must include clear docstrings, full type hints, and overloads/generics where appropriate.

Type checkers/python typing limitations means we need sometimes to do things that are very "dumb" (see the signature of `Iter::flatten` for yourself).

#### Performance-critical code

Internal implementations must prioritize performance and don't need docstrings.

We prefer "vanilla" Python code instead of our own API to maximize performance, i.e using for loops, comprehensions and built-in functions.

We do it here so we don't have to do it anymore anywhere we use pyochain :-).

We want to use each trick possible, for example `fn = object.method` and then `fn()` instead of `object.method()`, to avoid attribute lookups.

#### Where should a method live?

pyochain provides ABCs and concrete types. We always want to add methods to ABCs whenever possible, but this can be do only in some cases:

- In any case, it must not depend on anything else than what is already available in the dunders.
- If it's an aggregate, or it returns `None` because of mutation. e.g `Pyoiterable::sum` or `PyoMutableSequence::retain`.
- If it goes from collection -> iterator or the other way around. e.g `PyoIterable::iter` or `PyoIterator::{sort, tail, try_collect}`.

The general idea being that if a method return an object from the same family (e.g a `PyoMutableSequence` that need to do operations to ultimately return a `Vec`), then it should be in the concrete class, because this would entail confusion: We have two similar data structures, why are we swapping from one to the other here? Especially an abstract one to a concrete one.

On the other hand, if we go from a `PyoSequence` to an `Iter`, then it makes more sense to have it in the ABC, because it's a common operation that doesn't depend on the internal implementation of the collection, and goes from one structure to another.

#### Docstrings

docstrings should follow the format below.

Note that blank line between the example result and the end of the code block.

Doctest will fail otherwise, because it will consider the "```" as part of the output.

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

For development mode (fast compile, no optimizations):

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
uv run maturin develop --uv
```

## Tests and quality checks

Before committing, ensure all checks pass.

### type checking/linting/formatting

```bash
uv run ruff check . --fix --unsafe-fixes;
uv run ruff format . --preview;
uv run basedpyright src/pyochain;
uv run -m scripts.check_docstrings;
uv run pydoclint src/pyochain;
uv run -m scripts.generate_docs
```

Unfortunately, `Ruff` doesn't work well when doctests are mixed with backticks sections in docstrings to format code examples.

The workaround is to temporarily remove them, run `Ruff` and then put them back.

For multiple sections, you can use your IDE to replace both of them by dummy text, run `Ruff` and then replace the dummy text by the original backticks.

### tests

Add `--cov=src --cov-report=term-missing` to the pytest command below to include coverage reports.

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
