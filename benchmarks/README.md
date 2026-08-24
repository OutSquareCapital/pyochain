# Benchmarks

This folder contains `pytest-benchmark`-based micro-benchmarks for the public
Python API.

## Useful commands

Run only benchmarks:

```powershell
uv run pytest benchmarks/ --benchmark-only
```

Group output by parametrized dataset size:

```powershell
uv run pytest benchmarks/ --benchmark-only --benchmark-group-by=param:size
```

Save a run with a readable name:

```powershell
uv run pytest benchmarks/ --benchmark-only --benchmark-save=iter-sizes
```

Save stats plus raw timing data:

```powershell
uv run pytest benchmarks/ --benchmark-only --benchmark-save-data --benchmark-autosave
```

Compare against the latest saved run:

```powershell
uv run pytest benchmarks/ --benchmark-only --benchmark-compare
```

Advanced example:

```shell
uv run pytest benchmarks/test_sequences.py::test_init --benchmark-only --benchmark-group-by=param:size --benchmark-group-by=name --benchmark-warmup=true --benchmark-disable-gc --benchmark-columns=median,mean,min,max,stddev --benchmark-sort=mean --benchmark-compare
```

Run a single test, grouped by size + name, compare against the last saved run, with warmup and GC disabled, and show only the median, mean, min, max, and stddev columns.

Also:

```shell
uv run pytest benchmarks/test_sequences.py::test_init -k "10-"--benchmark-only --benchmark-group-by=name --benchmark-warmup=true --benchmark-disable-gc --benchmark-columns=median,mean,min,max,stddev --benchmark-sort=mean --benchmark-compare
```

Here we add a `k` filter to only run the 10- element size, and group by name only.

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

## Import benchmark

If you want to check import speed, you can use the builtin python command:

```powershell
uv run python -X importtime -c "import pyochain"
```

To get a table with sorted import times:

```powershell
uv run python -X importtime -c "import pyochain" 2>&1 |
Select-String "import time:" |
Where-Object { $_ -notmatch "cumulative" } |
ForEach-Object {
    $parts = $_ -split '\|'
    [PSCustomObject]@{
        Self = [int](($parts[0] -replace '.*:').Trim())
        Cumulative = [int]($parts[1].Trim())
        Module = $parts[2].Trim()
    }
} |
Sort-Object Cumulative -Descending |
Format-Table -AutoSize
```

## Sources

- Docs overview: <https://pytest-benchmark.readthedocs.io/en/latest/>
- Usage and CLI options: <https://pytest-benchmark.readthedocs.io/en/latest/usage.html>
- Comparing saved runs: <https://pytest-benchmark.readthedocs.io/en/latest/comparing.html>
- FAQ on noisy results: <https://pytest-benchmark.readthedocs.io/en/latest/faq.html>
- Upstream source: <https://github.com/ionelmc/pytest-benchmark>
