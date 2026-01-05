# Cookbook

This cookbook provides practical examples of how to use the `pyochain` library for various data manipulation tasks in Python.
Each example demonstrates a specific use case, showcasing the power and flexibility of `pyochain` for functional programming and data processing.

## Combining Option, Result and Iterators in Data Pipelines

Classes have been designed to work seamlessly together, enabling complex data processing pipelines with clear error handling.
**Note**: We return pc.Ok(None) for simplicity, the commented line shows how you would use it in practice.

```python
>>> from pathlib import Path
>>> import polars as pl
>>> import pyochain as pc
>>> def safe_parse_int(s: str) -> pc.Result[int, str]:
...     try:
...         return pc.Ok(int(s))
...     except ValueError:
...         return pc.Err(f"Invalid integer: {s}")
>>>
>>> def _write_to_file(lf: pl.LazyFrame, file_path: Path) -> pc.Result[None, str]:
...     """Write the LazyFrame to a CSV file."""
...     try:
...         # lf.filter(pl.col("value").gt(15)).sink_parquet(file_path)
...         return pc.Ok(None)
...     except (OSError, pl.exceptions.ComputeError) as e:
...         return pc.Err(f"Failed to write to file: {e}")
>>> PATH = Path("parsed_integers.parquet")
>>> data = ["10", "20", "foo", "30", "bar"]
>>> results = (
...    pc.Iter(data)
...    .map(safe_parse_int)  # Parse each string safely
...    .filter_map(lambda r: r.ok())  # Keep only successful parses
...    .enumerate()  # Add indices
...    .collect()  # Materialize the results
...    .inspect(
...        lambda seq: print(f"Parsed integers: {seq}") # Log parsed integers
...    )
...    .into(pl.LazyFrame, schema=["index", "value"])  # Pass to Polars LazyFrame
...    .pipe(_write_to_file, PATH)  # Write to file
...    .map_err(lambda e: print(f"Error: {e}"))  # Print error message
... )
Parsed integers: Seq((0, 10), (1, 20), (2, 30))
>>> results
Ok(None)

```

## Finding the Earliest Compatible Dependency Versions

**Problem:**
Multiple folders in a src tree may have their own `requirements.txt` files specifying dependencies with version constraints.
e.g:

```text
--- src/my_project/module_a/requirements.txt
  numpy>=1.20.0
  pandas==1.3.0
--- src/my_project/module_b/requirements.txt
  numpy>=1.18.0
    pandas>=1.2.0
etc...
```

**Goal:**
To find all the files paths, and then their earliest compatible versions for each dependency across all files.

**Solution:**

```python
from enum import StrEnum, auto
from pathlib import Path

import pyochain as pc

PROJECT = "my_project"
SRC = Path("src").joinpath(PROJECT)


class Splitter(StrEnum):
    EQ = "=="
    GT = ">="
    LT = "<="
    UNSPECIFIED = auto()


def _split_version(line: str, pos: int) -> str:
    def _split_with(splitter: Splitter) -> str:
        return line.split(splitter)[pos]

    return (
        _split_with(Splitter.EQ)
        if Splitter.EQ in line
        else _split_with(Splitter.GT)
        if Splitter.GT in line
        else _split_with(Splitter.LT)
        if Splitter.LT in line
        else Splitter.UNSPECIFIED
    )


def find_paths() -> pc.Seq[Path]:
    """Find all requirements.txt files in the src/project directory."""
    return pc.Iter(SRC.rglob("*requirements.txt")).collect()


def main() -> pc.Dict[str, str]:
    return (
        pc.Iter(SRC.rglob("*requirements.txt"))
        .map(lambda p: p.read_text().splitlines())
        .flatten()
        .group_by(lambda line: _split_version(line, 0))
        .map(
            lambda lines: (
                lines.key,
                lines.values.map(lambda line: _split_version(line, 1).strip())
                .sort()
                .first(),
            )
        )
        .collect(pc.Dict)
    )


if __name__ == "__main__":
    find_paths()
    main()

```

### Determining All Public Methods of a Class

Below is an example of using pyochain to extract and enumerate all public methods of a class.

```python
>>> import pyochain as pc
>>> def get_public_methods(cls: type) -> dict[int, str]:
...     return (
...         pc.Iter(cls.mro())
...         .flat_map(lambda x: x.__dict__.values())
...         .filter(lambda f: callable(f) and not f.__name__.startswith("_"))
...         .map(lambda f: f.__name__)
...         .sort()
...         .iter()
...         .enumerate()
...         .collect(dict)
...     )
>>> methods = get_public_methods(pc.Iter)
>>> "map" in methods.values() and "filter" in methods.values()
True

```

For comparison, here's the equivalent using pure Python:

```python
>>> import pyochain as pc
>>> def get_public_methods_pure(cls: type) -> dict[int, str]:
...     dict_values = []
...     for klass in cls.mro():
...         dict_values.extend(klass.__dict__.values())
...     return dict(enumerate(sorted([
...         obj.__name__
...         for obj in dict_values
...         if callable(obj) and not obj.__name__.startswith("_")
...     ])))
>>> methods_pure = get_public_methods_pure(pc.Iter)
>>> "map" in methods_pure.values() and "filter" in methods_pure.values()
True

```
