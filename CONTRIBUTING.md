# Contributing to pyochain

Thank you for your interest in contributing to pyochain!

This document covers environment setup, the commands to run, and how to commit and release.

For the architecture and coding conventions, see [AGENTS](./AGENTS.md).

It is destined for any developer, wether human or machine.

## Setup

This project uses `uv` to manage everything python-related.

After cloning the repo, copy the [cargo config example](./.cargo/config.toml.example) to `.cargo/config.toml` (git-ignored) and adapt it to your platform to set-up the python path for PyO3.

Then you can sync the venv with `uv`.

`--all-groups` will also install the dependencies necessary for the website documentation, and the type-checking tests.

```bash
uv sync --dev
uv sync --all-groups
```

To set clippy and per-crate compilation settings for `rust-analyzer`, a [`zed`](.zed/settings.json) and [`vscode`](.vscode/settings.json) config files are present.

Open for PR's for linux/mac cargo config examples, as well as others editors configs.

### Usual workflow

The command you will run most often is the following, which will build the package in development mode and run the tests:

```bash
uv run maturin develop --uv;
uv run pytest
```

For benchmarks, you can run the following command:

```bash
uv run maturin develop --uv --profile profiling
```

See [the readme](./benchmarks/README.md) for more informations.

## linting/formatting/type checking

Before any pull request, or commit to the master branch, you need to ensure that all checks pass.

Run the following command for more informations:

```bash
uv run -m scripts ci --help
```

## Documentation

### Stubs Docstrings

Docstrings should follow the google format. See more information on [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).

The code in the `examples` section will be automatically part of the test suite (`pytest-docflex`, which also collects the [`README`](README.md) and [`docs`](./docs/)).

Every public function/method in a stub needs a docstring with at least one closed `python` code block.

We use code blocks instead of doctests, which means that the whole pytest ecosystem is available, which can be handy for expected failures.

See [this file](./pyochain/abc/_iterator.pyi) for a practical reference for documentation style.

### Website build

To build and serve the documentation locally, run the command below.

Note that the `-c` flag is necessary, has `zensical` still has various issues with caching and will give inconsistent results with it.

```bash
uv run zensical build -c
```

Then open your browser with the [site](site/index.html) to view the generated documentation.

## Contributing workflow

- Create a branch per feature/fix and keep commits focused and descriptive.
- Run all quality checks locally before opening a pull request.
- Include tests or doctest examples for behavior changes whenever possible.
- For Rust changes, consider adding benchmarks to verify performance impact when pertinent.

This project follow the [gitmoji convention](https://gitmoji.dev/) for commit messages and release notes sections.

Publishing a GitHub release triggers two workflows:

- [`publish`](.github/workflows/publish.yml) => publish the package on Pypi.
- [`docs`](.github/workflows/docs.yml) => builds the website and deploys it to GitHub Pages.

The version to release is the one in `pyproject.toml`.

### Issue on release

If an issue on a release appear, AND the package is NOT published on Pypi, running the following commands can help going back to a clean state without needing to create a new release:

```bash
git tag -d <tag_name>
git push origin --delete <tag_name>
```

This will convert the last tag into a draft release, allowing you to fix the issue and publish the release again without creating a new one.
