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
SRC = Path(f"src/{PROJECT}")

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


def find_paths():
    """Find all requirements.txt files in the src/project directory."""
    return (
        pc.Iter(SRC.rglob("*requirements.txt"))
        .collect()
        .println()
    )

def main():
    return (
        pc.Iter(SRC.rglob("*requirements.txt"))
        .map(lambda p: p.read_text().splitlines())
        .flatten()
        .group_by(lambda line: _split_version(line, 0))
        .map_values(
            lambda lines: pc.Seq(lines)
            .iter()
            .map(lambda line: _split_version(line, 1).strip())
            .sort()
            .first()
        )
        .println()
    )


if __name__ == "__main__":
    find_paths()
    main()
```

## Getting Plotly Color Palettes

In one of my project, I have to introspect some modules from plotly to get some lists of colors.

I want to check wether the colors are in hex format or not, and I want to get a dictionary of palettes.

The script below shows you can smoothly interoperate between pyochain and polars to achieve this in a readable way.

```python

from types import ModuleType

import polars as pl
import pyochain as pc
from plotly.express.colors import cyclical, qualitative, sequential



MODULES: set[ModuleType] = {
    sequential,
    cyclical,
    qualitative,
}

def get_palettes() -> pc.Dict[str, list[str]]:
    clr = "color"
    scl = "scale"
    df: pl.DataFrame = (
        pc.Iter.from_(MODULES)
        .map(
            lambda mod: pc.Dict.from_object(mod)
            .filter_values(lambda v: isinstance(v, list))
            .unwrap()
        )
        .into(pl.LazyFrame)
        .unpivot(value_name=clr, variable_name=scl)
        .drop_nulls()
        .filter(
            pl.col(clr)
            .list.eval(pl.element().first().str.starts_with("#").alias("is_hex"))
            .list.first()
        )
        .sort(scl)
        .collect()
    )
    keys: list[str] = df.get_column(scl).to_list()
    values: list[list[str]] = df.get_column(clr).to_list()
    return pc.Iter.from_(keys).with_values(values)


# Ouput excerpt:
{'mygbm_r': ['#ef55f1',
            '#c543fa',
            '#9139fa',
            '#6324f5',
            '#2e21ea',
            '#284ec8',
            '#3d719a',
            '#439064',
            '#31ac28',
            '#61c10b',
            '#96d310',
            '#c6e516',
            '#f0ed35',
            '#fcd471',
            '#fbafa1',
            '#fb84ce',
            '#ef55f1']}
```

However you can still easily go back with for loops when the readability is better this way.

In another place, I use this function to generate a Literal from the keys of the palettes.

```python

from enum import StrEnum

class Text(StrEnum):
    CONTENT = "Palettes = Literal[\n"
    END_CONTENT = "]\n"
    ...# rest of the class

def generate_palettes_literal() -> None:
    literal_content: str = Text.CONTENT
    for name in get_palettes().iter_keys().sort().unwrap():
        literal_content += f'    "{name}",\n'
    literal_content += Text.END_CONTENT
    ...# rest of the function
```

Since I have to reference the literal_content variable in the for loop, This is more reasonnable to use a for loop here rather than a map + reduce approach.

### Determining All Public Methods of a Class

Below is an example of using pyochain to get all the public methods of the `pc.Iter` class, both with pyochain and with pure python.

```python
from collections.abc import Sequence
from typing import Any

import pyochain as pc


def get_all_iter_methods() -> Sequence[tuple[int, str]]:
    return (
        pc.Seq(pc.Iter.mro())
        .iter()
        .map(lambda x: x.__dict__.values())
        .flatten()
        .map_if(
            predicate=lambda f: callable(f) and not f.__name__.startswith("_")
        )
        .then(lambda f: f.__name__)
        .sort()
        .iter()
        .enumerate()
        .collect()
        .inner()
    )


def get_all_iter_methods_pure_python() -> list[tuple[int, str]]:
    dict_values: list[Any] = []
    for cls in pc.Iter.mro():
        dict_values.extend(cls.__dict__.values())

    return list(
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
```

Output excerpt, if returning immediately after collect, and then calling println():

```text
PS C:\Users\tibo\python_codes\pyochain> uv run foo.py
[(0, 'accumulate'),
 (1, 'adjacent'),
 (2, 'all'),
 (3, 'all_equal'),
 (4, 'all_unique'),
 (5, 'any'),
 (6, 'apply'),
 (7, 'apply'),
 ...
]
```
