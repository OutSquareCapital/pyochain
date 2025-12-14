# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

`pyochain` brings a fluent, declarative API inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like Polars to your everyday Python iterables and dictionaries.

Manipulate data through composable chains of operations and manage errors and optional values safely, all while enjoying type-safe guarantees.

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

* **Rust's language and  Rust stdlib:** Emulate naming conventions (`from_()`, `into()`) and leverage concepts from Rust's powerful iterator traits (method chaining, lazy evaluation), Option and Result enums, to bring similar expressiveness to Python.
* **Python iterators libraries:** Libraries like `rolling`, `cytoolz`, and `more-itertools` provided ideas, inspiration, and implementations for many of the iterator methods.
* **PyFunctional:** Although not directly used (because I started writing pyochain before discovering it), also shares similar goals and ideas.

### Core Components

#### `Iter[T]`

A superset of Iterator. All operations are **lazy**, consuming the underlying iterator only when needed.

This allows for efficient processing of large or even infinite sequences.

To create an `Iter`, you can:

* Take any iterable: `pc.Iter(my_iterable)`
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

### Core Piping Methods

All wrappers provide a small set of core methods for chaining and passing values to your own functions.

This is covered in detail in the guide: see
[`Guides → Chaining Methods`](docs/guides/chaining.md).

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
