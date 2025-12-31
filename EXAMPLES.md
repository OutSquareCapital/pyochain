# Cookbook

This cookbook provides practical examples of how to use the `pyochain` library for various data manipulation tasks in Python.
Each example demonstrates a specific use case, showcasing the power and flexibility of `pyochain` for functional programming and data processing.

## Finding the Earliest Compatible Dependency Versions

Problem:
Multiple folders in a src tree may have their own `requirements.txt` files specifying dependencies with version constraints.
To find all the files paths, and then their earliest compatible versions, you can use the following script:

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
