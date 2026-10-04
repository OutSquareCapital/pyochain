# Contributing to pyochain

Thank you for your interest in contributing to pyochain!

This document outlines the repository structure, coding standards, and contribution workflow to help you get started.

## Repository overview

NOTE: The project evolve quickly, so this section is often outdated. It can be useful as a rough idea when first exploring the repository, but for accurate information, refer to the actual source code, or raise an issue if you find something unclear/deserves to be clearly documented.

### Python API and typing

All the stubs are located in the `pyochain` folder.

The stub packages follow the public Rust module hierarchy, but the mapping is not strictly one-to-one: package initializers, grouped stubs, and private Rust helper modules do not always have a matching file.

### Rust and PyO3 implementation

The actual source code implementation lives in the `src` folder, with the following structure:

- [src/lib.rs](src/lib.rs) — initializes the `pyochain` PyO3 module and registers the `core`, `abc`, `collections`, and `collections._sorted` submodules.
- [src/core/](src/core/) — implements the core types.
- [src/abc/](src/abc/) — implements the abstract base classes, mixins, and shared ABC traits.
- [src/collections/](src/collections/) — implements concrete collections such as `Deque`, `HeapMax`, `HeapMin`, `StableSet` etc...
- [src/collections/sorted/](src/collections/sorted/) — implements sorted collections, views, iterators, and their internal support modules.
- [src/traits.rs](src/traits.rs) — defines shared wrapper, conversion, and initialization traits.

### Internal crates

- [crates/pyo3_ext/](crates/pyo3_ext/) — internal PyO3 extensions and utility traits.
- [crates/pyochain_macros/](crates/pyochain_macros/) — procedural macros used by the Rust implementation.
- [crates/pyochain_build/](crates/pyochain_build/) — build tool for generating documentation and validating the repository.

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

## Setup

This project uses `uv` to manage everything python-related.

After cloning the repo, adapt the [cargo config example](.cargo\config.toml.example) to your platform specifics to set-up the python path for it.

Then you can sync the venv with `uv`.

`--all-groups` will also install the dependencies necessary for the website documentation.

```bash
uv sync --dev
uv sync --all-groups
```

A `zed` config file is present to set clippy and per-crate compilation for `rust-analyzer` for more convenient local development.

I don't use `VScode` anymore but it should be easy to set up a similar configuration.

Open for PR's for linux/mac cargo config examples, as well as others editors configs.

### Usual workflow

The command you will run most often is the following, which will build the package in development mode and run the tests:

```bash
uv run maturin develop --uv;
uv run pytest
```

## linting/formatting/type checking

Before any pull request, or commit to the master branch, you need to ensure that all checks pass. Run the following command for more informations:

```bash
uv run -m scripts ci --help
```

## Documentation

### Stubs Docstrings

docstrings should follow the google format. See more information on [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

The code in the `examples` section will be automatically part of the test suite.

We use code blocks instead of doctests, which means that the whole pytest ecosystem is available, which can be handy for expected failures.

See [this file](pyochain/abc/_iterator.pyi) for a practical reference for documentation style.

### Automatic generation

Prior to a release, to check correct documentation generation, or to build the tool itself, you can run one of the following commands:

```bash
cargo run -p pyochain-build
cargo run --release -p pyochain-build
cargo build -p pyochain-build
cargo build --release -p pyochain-build
```

### Website build

To build and serve the documentation locally, run the command below.

Note that the `-c` flag is necessary, has `zensical` still has various issues with caching and will give inconsistent results with it.

```bash
uv run zensical build -c
```

Then open your browser with the [site](site/index.html) to view the generated documentation.

### Benchmarks

See [the readme](benchmarks/README.md) for more information on running and saving benchmarks.

## Contributing workflow

- Create a branch per feature/fix and keep commits focused and descriptive.
- Run all quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
- For Rust changes, consider adding benchmarks to verify performance impact when pertinent.

Each commit should be prefixed with one of the following tags:

- `enh` => enhancement, improved typing, API documentation, etc...
- `fix` => bug fix, logical error correction, typo, etc...
- `refactor` => code refactoring, no functional change
- `feat` => new feature
- `chore` => maintenance task, CI, build, dev documentation, etc...
- `perf` => performance improvement, no behavior change

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
