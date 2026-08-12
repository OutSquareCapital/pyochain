# Contributing to pyochain

Thank you for your interest in contributing to pyochain! This document outlines the repository structure, coding standards, and contribution workflow to help you get started.
nchmarks.

## Repository overview

### Python API and typing

- [pyochain/](pyochain/) — Python stubs for the public API. They match their Rust modules 1:1 once your strip their "_" prefix.
Only the two following files do not have a corresponding Rust module
- [pyochain/_types.pyi](pyochain/_types.pyi) — shared typing protocols and type aliases.
- [pyochain/pyochain.pyi](pyochain/pyochain.pyi) — public re-exports. You could maybe say it corresponds to the `src/lib.rs` Rust module.

### Rust and PyO3 implementation

- [src/lib.rs](src/lib.rs) — initializes the `pyochain` PyO3 module and registers its public classes, functions, ABCs, and collection submodules.
- [src/dict.rs](src/dict.rs) — implements `Dict`.
- [src/display.rs](src/display.rs) — formats Python objects for `repr` output.
- [src/errors.rs](src/errors.rs) — defines the Python exceptions raised by failed `Option` and `Result` unwraps.
- [src/hasher.rs](src/hasher.rs) — provides the hash helper used by `Option` and `Result` values.
- [src/iterators.rs](src/iterators.rs) — implements `Iter`, `Peekable`, and the iterator adapters exposed by `pyochain._iterators`.
- [src/option.rs](src/option.rs) — implements `Option`, `Some`, `Null`, `NONE`, and the related helper functions.
- [src/pyovec.rs](src/pyovec.rs) — implements `Vec`.
- [src/range.rs](src/range.rs) — implements `Range`.
- [src/result.rs](src/result.rs) — implements `Result`, `Ok`, and `Err`.
- [src/seq.rs](src/seq.rs) — implements `Seq`.
- [src/sets.rs](src/sets.rs) — implements `Set` and `SetMut`.
- [src/sliceview.rs](src/sliceview.rs) — implements `SliceView` and its slice iterators.
- [src/traits.rs](src/traits.rs) — defines the shared wrapper, conversion, and ABC initialization traits.
- [src/abc/](src/abc/) — abstract base classes, traits, and mixins.
- [src/collections/](src/collections/) — concrete collection implementations, including sorted collections.

### Internal crates

- [crates/pyo3_ext/](crates/pyo3_ext/) — internal PyO3 extensions and utility traits.
- [crates/pyochain_macros/](crates/pyochain_macros/) — procedural macros used by the Rust implementation.

### Tests, documentation, and tooling

- [tests/](tests/) — Python tests, ABC tests, external integration tests, and benchmarks.
- [docs/](docs/) — documentation sources and API reference pages.
- [scripts/](scripts/) — documentation generation and repository validation scripts.
- [Cargo.toml](Cargo.toml) — Rust workspace and dependency configuration.
- [pyproject.toml](pyproject.toml) — Python package metadata, maturin configuration, and development dependencies.
- [pyrefly.toml](pyrefly.toml) — Pyrefly configuration.
- [ruff.toml](ruff.toml) — Ruff linting and formatting configuration.
- [zensical.toml](zensical.toml) — documentation site configuration.

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
uv run -m scripts.check_nav
```

Unfortunately, `Ruff` doesn't work well when doctests are mixed with backticks sections in docstrings to format code examples.

The workaround is to temporarily remove them, run `Ruff` and then put them back.

For multiple sections, you can use your IDE to replace both of them by dummy text, run `Ruff` and then replace the dummy text by the original backticks.

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
