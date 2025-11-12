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

A wrapper for any `Iterator` or `Generator`. All operations are **lazy**, consuming the underlying iterator only when needed.

This allows for efficient processing of large or even infinite sequences.

To create an `Iter`, you can:

* Wrap an existing iterator/generator: `pc.Iter(my_iterator)`
* Convert any iterable: `pc.Iter.from_(my_list)`
* Wrap unpacked values: `pc.Iter.from_(1, 2, 3)`
* Use built-in constructors like `pc.Iter.from_count()` for infinite sequences.

#### `Seq[T]`

A wrapper for a `Sequence` (like a `list` or `tuple`), representing an **eagerly** evaluated collection of data.
`Seq` is useful when you need to store results in memory, access elements by index, or reuse the data multiple times.

It shares many methods with `Iter` but performs operations immediately.
You can switch between lazy and eager evaluation by using `my_seq.iter()` and `my_iter.collect()`.

#### `Dict[K, V]`

A wrapper for a `dict`, providing a rich, chainable API for dictionary manipulation. It simplifies common tasks like filtering, mapping, and transforming dictionary keys and values.

Key features include:

* **Immutability**: Most methods return a new `Dict` instance, preventing unintended side effects.
* **Nested Data Utilities**: Easily work with complex, nested dictionaries using methods like `pluck` and `flatten`.
* **Flexible Instantiation**: Create a `Dict` from mappings, iterables of pairs, or even object attributes with `Dict.from_object()`.

#### `Result[T, E]`

A type for functions that can fail, inspired by Rust's `Result`. It represents either a success (`Ok[T]`) containing a value or an error (`Err[E]`) containing an error. It forces you to handle potential failures explicitly, leading to more robust code.

#### `Option[T]`

A type for values that may be absent, inspired by Rust's `Option`. It represents either the presence of a value (`Some[T]`) or its absence (`NONE`). It provides a safe and expressive way to handle optional values without resorting to `None` checks everywhere.

#### `Wrapper[T]`

A generic wrapper for any Python object, allowing it to be integrated into a `pyochain` fluent-style chain. Use `Wrapper` to `pipe`, `apply`, or `into` functions when working with objects that don't have their own `pyochain` wrapper, such as instances of custom classes or third-party library objects.

### Core Piping Methods

All wrappers provide a set of common methods for chaining and data manipulation:

* `into(func, *args, **kwargs)`: Passes the **unwrapped** data to `func` and returns the raw result. This is a terminal operation that ends the chain.
* `apply(func, *args, **kwargs)`: Passes the **unwrapped** data to `func` and **re-wraps** the result in the same wrapper type for continued chaining.
* `pipe(func, *args, **kwargs)`: Passes the **wrapped instance** (`self`) to `func`. This allows you to insert custom functions into the chain that operate on the wrapper itself.
* `println()`: Prints the unwrapped data to the console for debugging and returns `self` to continue the chain.
* `inner()`: Returns the underlying wrapped data.

### Rich Lazy Iteration (`Iter`)

Leverage dozens of methods inspired by Rust's `Iterator`, `itertools`, `cytoolz`, and `more-itertools`.

```python
import pyochain as pc

result = (
    pc.Iter.from_count(1)  # Infinite iterator: 1, 2, 3, ...
    .filter(lambda x: x % 2 != 0)  # Keep odd numbers
    .map(lambda x: x * x)  # Square them
    .take(5)  # Take the first 5
    .into(list)  # Consume into a list
)
# result: [1, 9, 25, 49, 81]
```

### Type-Safe Error Handling (`Result` and `Option`)

Write robust code by handling potential failures explicitly.

```python
import pyochain as pc

def divide(a: int, b: int) -> pc.Result[float, str]:
    if b == 0:
        return pc.Err("Cannot divide by zero")
    return pc.Ok(a / b)

# --- With Result ---
res1 = divide(10, 2)  # Ok(5.0)
res2 = divide(10, 0)  # Err("Cannot divide by zero")

# Safely unwrap or provide a default
value = res2.unwrap_or(0.0)  # 0.0

# Map over a successful result
squared = res1.map(lambda x: x * x)  # Ok(25.0)

# --- With Option ---
def find_user(user_id: int) -> pc.Option[str]:
    users = {1: "Alice", 2: "Bob"}
    return pc.Some(users.get(user_id)) if user_id in users else pc.NONE

user = find_user(1).map(str.upper).unwrap_or("Not Found")  # "ALICE"
not_found = find_user(3).unwrap_or("Not Found")  # "Not Found"
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
