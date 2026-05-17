# Benchmarks

This folder contains `pytest-benchmark`-based micro-benchmarks for the public
Python API.

## Useful commands

Run only benchmarks:

```powershell
uv run pytest tests/benchmarks --benchmark-only
```

Group output by parametrized dataset size:

```powershell
uv run pytest tests/benchmarks --benchmark-only --benchmark-group-by=param:size
```

Save a run with a readable name:

```powershell
uv run pytest tests/benchmarks --benchmark-only --benchmark-save=iter-sizes
```

Save stats plus raw timing data:

```powershell
uv run pytest tests/benchmarks --benchmark-only --benchmark-save-data --benchmark-autosave
```

Compare against the latest saved run:

```powershell
uv run pytest tests/benchmarks --benchmark-only --benchmark-compare
```

## Useful options

- `--benchmark-min-time`: minimum time per round
- `--benchmark-max-time`: max total time per benchmark
- `--benchmark-min-rounds`: minimum number of rounds
- `--benchmark-warmup` and `--benchmark-warmup-iterations`: warmup tuning
- `--benchmark-disable-gc`: remove GC noise during measurement
- `--benchmark-group-by=param:size`: useful when benchmarking several sizes
- `--benchmark-time-unit=ns|us|ms|s`: force display units
- `--benchmark-sort=min|mean|median|...`: choose table ordering

If you need fully fixed iterations and rounds instead of calibration, use
`benchmark.pedantic(...)`.

## Saved format

Saved runs go under `.benchmarks/<platform-python>/` by default, for example:

```text
.benchmarks/Windows-CPython-3.14-64bit/0001_iter-sizes.json
```

or with autosave:

```text
.benchmarks/Windows-CPython-3.14-64bit/0001_<commit>_<timestamp>.json
```

- `--benchmark-save` and `--benchmark-autosave` save JSON benchmark reports
- `--benchmark-save-data` includes raw timing samples in those JSON files
- `--benchmark-json path.json` writes a full JSON report to a path you choose

## Sources

- Docs overview: <https://pytest-benchmark.readthedocs.io/en/latest/>
- Usage and CLI options: <https://pytest-benchmark.readthedocs.io/en/latest/usage.html>
- Comparing saved runs: <https://pytest-benchmark.readthedocs.io/en/latest/comparing.html>
- FAQ on noisy results: <https://pytest-benchmark.readthedocs.io/en/latest/faq.html>
- Upstream source: <https://github.com/ionelmc/pytest-benchmark>
