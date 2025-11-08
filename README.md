# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

`pyochain` brings a fluent, declarative API inspired by Rust's `Iterator` and DataFrame libraries like Polars to your everyday Python iterables and dictionaries.

Manipulate data through composable chains of operations, enhancing readability and reducing boilerplate.

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

## Installation

```bash
uv add pyochain
```

## API Reference 📖

The full API reference can be found at:
<https://outsquarecapital.github.io/pyochain/>

## Overview

### Philosophy

* **Declarative over Imperative:** Replace explicit `for` and `while` loops with sequences of high-level operations (map, filter, group, join...).
* **Fluent Chaining:** Each method transforms the data and returns a new wrapper instance, allowing for seamless chaining.
* **Lazy and Eager:** `Iter` operates lazily for efficiency on large or infinite sequences, while `Seq` represents materialized sequences for eager operations.
* **100% Type-safe:** Extensive use of generics and overloads ensures type safety and improves developer experience.
* **Documentation-first:** Each method is thoroughly documented with clear explanations, and usage examples. Before any commit is made, each docstring is automatically tested to ensure accuracy. This also allows for a convenient experience in IDEs, where developers can easily access documentation with a simple hover of the mouse.
* **Functional paradigm:** Design encourages building complex data transformations by composing simple, reusable functions on known buildings blocks, rather than implementing customs classes each time.

### Inspirations

* **Rust's language and  Rust `Iterator` Trait:** Emulate naming conventions (`from_()`, `into()`) and leverage concepts from Rust's powerful iterator traits (method chaining, lazy evaluation) to bring similar expressiveness to Python.
* **Python iterators libraries:** Libraries like `rolling`, `cytoolz`, and `more-itertools` provided ideas, inspiration, and implementations for many of the iterator methods.
* **PyFunctional:** Although not directly used (because I started writing pyochain before discovering it), also shares similar goals and ideas.

### Core Components

#### `Iter[T]`

To instantiate it, wrap a Python `Iterator` or `Generator`, or take any Iterable (`list`, `tuple`, etc...) and call Iter.from_ (which will call the builtin `iter()` on it).

All operations that return a new `Iter` are **lazy**, consuming the underlying iterator on demand.

Provides a vast array of methods for transformation, filtering, aggregation, joining, etc..

#### `Seq[T]`

Wraps a Python `Sequence` (`list`, `tuple`...), and represents **eagerly** evaluated data.

Exposes a subset of the `Iter` methods who operate on the full dataset (e.g., `sort`, `union`) or who aggregate it.

It is most useful when you need to reuse the data multiple times without re-iterating it.

Use `.iter()` to switch back to lazy processing.

#### `Dict[K, V]`

Wraps a Python `dict` (or any Mapping via ``Dict.from_``) and provides chainable methods specific to dictionaries (manipulating keys, values, items, nesting, joins, grouping).

Promote immutability by returning new `Dict` instances on each operation, and avoiding in-place modifications.

Can work easily on known data structure (e.g `dict[str, int]`), with methods like `map_values`, `filter_keys`, etc., who works on the whole `dict` in a performant way, mostly thanks to `cytoolz` functions.

But `Dict` can work also as well as on "irregular" structures (e.g., `dict[Any, Any]`, TypedDict, etc..), by providing a set of utilities for working with nested data, including:

* `pluck` to extract multiple fields at once.
* `flatten` to collapse nested structures into a single level.

#### `Wrapper[T]`

A generic wrapper for any Python object, allowing integration into `pyochain`'s fluent style using `pipe`, `apply`, and `into`.

Can be for example used to wrap numpy arrays, json outputs from requests, or any custom class instance, as a way to integrate them into a chain of operations, rather than breaking the chain to reference intermediate variables.

### Core Piping Methods

All wrappers inherit from `CommonBase`:

* `into[**P, R](func: Callable[Concatenate[T, P]], *args: P.args, **kwargs: P.kwargs) -> R`
    Passes the **unwrapped** data to `func` and returns the raw result (terminal).
* `apply[**P, R](func: Callable[Concatenate[T, P]], *args: P.args, **kwargs: P.kwargs) -> "CurrentWrapper"[R]`
    Passes the **unwrapped** data to`func` and **re-wraps** the result for continued chaining.
* `pipe[**P, R](func: Callable[Concatenate[Self, P]], *args: P.args, **kwargs: P.kwargs) -> R`
    Passes the **wrapped instance** (`self`) to `func` and returns the raw result (can be terminal).
* `println()`
    Prints the unwrapped data and returns `self`.

### Rich Lazy Iteration (`Iter`)

Leverage dozens of methods inspired by Rust's `Iterator`, `itertools`, `cytoolz`, and `more-itertools`.

```python
import pyochain as pc

result = (
    pc.Iter.from_count(1) # Infinite iterator: 1, 2, 3, ...
    .filter(lambda x: x % 2 != 0) # Keep odd numbers: 1, 3, 5, ...
    .map(lambda x: x * x) # Square them: 1, 9, 25, ...
    .take(5) # Take the first 5: 1, 9, 25, 49, 81
    .into(list) # Consume into a list
)
# result: [1, 9, 25, 49, 81]
```

### Typing enforcement

Each method and class make extensive use of generics, type hints, and overloads (when necessary) to ensure type safety and improve developer experience.

Since there's much less need for intermediate variables, the developper don't have to annotate them as much, whilst still keeping a type-safe codebase.

### Convenience mappers: itr and struct

Operate on iterables of iterables or iterables of dicts without leaving the chain.

```python
import pyochain as pc

nested = pc.Iter.from_([[1, 2, 3], [4, 5]])
totals = nested.itr(lambda it: it.sum()).into(list)
# [6, 9]

records = pc.Iter.from_(
    [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
    ]
)
names = records.struct(lambda d: d.pluck("name").unwrap()).into(list)
# ['Alice', 'Bob']
```

## Key Dependencies and credits

Most of the computations are done with implementations from the `cytoolz`, `more-itertools`, and `rolling` libraries.

An extensive use of the `itertools` stdlib module is also to be noted.

pyochain acts as a unifying API layer over these powerful tools.

<https://github.com/pytoolz/cytoolz>

<https://github.com/more-itertools/more-itertools>

<https://github.com/ajcr/rolling>

The stubs used for the developpement, made by the maintainer of pyochain, can be found here:

<https://github.com/py-stubs/cytoolz-stubs>

---

## Real-life simple example

In one of my project, I have to introspect some modules from plotly to get some lists of colors.

I want to check wether the colors are in hex format or not, and I want to get a dictionary of palettes.
We can see here that pyochain allow to keep the same style than polars, with method chaining, but for plain python objects.

Due to the freedom of python, multiple paradigms are implemented across libraries.

If you like the fluent, functional, chainable style, pyochain can help you to keep it across your codebase, rather than mixing object().method().method() and then another where it's [[... for ... in ...] ... ].

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
