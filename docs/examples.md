# Cookbook

This cookbook provides practical examples of how to use the `pyochain` library for various data manipulation tasks in Python.
Each example demonstrates a specific use case, showcasing the power and flexibility of `pyochain` for functional programming and data processing.

## Basic Usage

### Chained Data Transformations

```python
>>> import pyochain as pc
>>>
>>> result = (
...    pc.Iter.from_count(1)  # Infinite iterator: 1, 2, 3, ...
...    .filter(lambda x: x % 2 != 0)  # Keep odd numbers
...    .map(lambda x: x * x)  # Square them
...    .take(5)  # Take the first 5
...    .collect()  # Materialize the result into a Seq
... )
>>> result
Seq(1, 9, 25, 49, 81)

```

### Type-Safe Error Handling (`Result` and `Option`)

Write robust code by handling potential failures explicitly.

```python
>>> import pyochain as pc
>>>
>>> def divide(a: int, b: int) -> pc.Result[float, str]:
...     if b == 0:
...         return pc.Err("Cannot divide by zero")
...     return pc.Ok(a / b)
>>>
>>> # --- With Result ---
>>> res1 = divide(10, 2)
>>> res1
Ok(5.0)
>>> res2 = divide(10, 0)
>>> res2
Err('Cannot divide by zero')
>>> # Safely unwrap or provide a default
>>> res2.unwrap_or(0.0)
0.0
>>> # Map over a successful result
>>> res1.map(lambda x: x * x)
Ok(25.0)
>>> # --- With Option ---
>>> def find_user(user_id: int) -> pc.Option[str]:
...     users = {1: "Alice", 2: "Bob"}
...     return pc.Some(users.get(user_id)) if user_id in users else pc.NONE
>>>
>>> find_user(1).map(str.upper).unwrap_or("Not Found")
'ALICE'
>>> find_user(3).unwrap_or("Not Found")
'Not Found'

```

### Combining them in Data Pipelines

Classes have been designed to work seamlessly together, enabling complex data processing pipelines with clear error handling.

```python
from pathlib import Path

import polars as pl

import pyochain as pc


def safe_parse_int(s: str) -> pc.Result[int, str]:
    try:
        return pc.Ok(int(s))
    except ValueError:
        return pc.Err(f"Invalid integer: {s}")


def _write_to_file(lf: pl.LazyFrame, file_path: Path) -> pc.Result[None, str]:
    """Write the LazyFrame to a CSV file."""
    try:
        return pc.Ok(lf.filter(pl.col("value") > 15).sink_parquet(file_path))
    except (OSError, pl.exceptions.ComputeError) as e:
        return pc.Err(f"Failed to write to file: {e}")


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
)

```

### Concrete Use Cases

Below are some more specific examples demonstrating how to use `pyochain` for specific tasks encountered in real-world scenarios.

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

Below is an example of using pyochain to get all the public methods of the `pc.Iter` class, both with pyochain and with pure python.

```python
from typing import Any

import pyochain as pc


def get_all_iter_methods() -> dict[int, str]:
    return (
        pc.Iter(pc.Iter.mro())
        .flat_map(lambda x: x.__dict__.values())
        .filter(lambda f: callable(f) and not f.__name__.startswith("_"))
        .map(lambda f: f.__name__)
        .sort()
        .iter()
        .enumerate()
        .collect(dict)
    )


def get_all_iter_methods_pure_python() -> dict[int, str]:
    dict_values: list[Any] = []
    for cls in pc.Iter.mro():
        dict_values.extend(cls.__dict__.values())

    return dict(
        enumerate(
            sorted(
                [
                    obj.__name__
                    for obj in dict_values
                    if callable(obj) and not obj.__name__.startswith("_")
                ],
            ),
        ),
    )


if __name__ == "__main__":
    methods = get_all_iter_methods()
    methods_pure = get_all_iter_methods_pure_python()
    assert methods == methods_pure

```

### Checking whether functions are implemented in Python or Rust

In this example, I want to check which functions of the Iter class or Rust Iterator class are not implemented in both languages.

It demonstrate how to combine pyochain with polars in uninterrupted data pipelines, who combines lazy Iterators, LazyFrames, and streaming mode writing.

```python
from typing import Literal

import polars as pl

import pyochain as pc


def _with_source(fn_name: str, src: Literal["python", "rust"]) -> tuple[str, str]:
    return (src, fn_name)


RUST_FN = [
    ...
]  # list of rust Iterator trait methods names as strings, copy pasted from website


def main() -> None:
    fn: pl.Expr = pl.col("fn")
    return (
        # create an iterator over the class hierarchy
        pc.Iter(pc.Iter.mro())
        # get the dict values view of each class and flatten them
        .flat_map(lambda x: x.__dict__.values())
        # keep only callables (methods of the classes here)
        .filter(callable)
        # get the method name, associate it with "python"
        .map(lambda x: _with_source(x.__name__, "python"))
        # do the same for rust functions, simply associating it with "rust"
        .chain(pc.Iter(RUST_FN).map(lambda x: _with_source(x, "rust")))
        # pass the iterator directly into a polars lazyframe, with specified schema (otherwise columns are named column_0, column_1)
        .into(lambda x: pl.LazyFrame(x, schema=["source", "fn"]))
        # keep only unique fn names, and those not starting with _ (dunder/private methods)
        .filter(fn.is_unique().and_(fn.str.starts_with("_").not_()))
        .sort(fn)  # sort by fn name
        # write the result to an ndjson file in streaming mode.
        .sink_ndjson("iter_fn_sources.ndjson")
    )

```
