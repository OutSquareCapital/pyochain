"""pyochain - A functional programming library for Python.

# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

`pyochain` brings a fluent, declarative API inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like Polars to your everyday Python iterables and dictionaries.

Manipulate data through composable chains of operations and manage errors and optional values safely, all while enjoying type-safe guarantees.

## Overview

Provides the following core classes and utilities:

- `Iter[T]`
  - A superset of Python `collections.abc.Iterator`, with chainable functional methods.
  - Underlying data structure is an `iterator` (if we can call it that).
  - Implement Iterator Protocol.
  - Can be converted to `Seq` with the `.collect()` method, or to `Vec` with `.collect_mut()` method.
- `Seq[T]`
  - An immutable collection with chainable methods.
  - Underlying data structure is a `tuple`.
  - Can be converted to `Iter` with the `.iter()` method.
  - Implement Sequence Protocol.
- `Vec[T]`
  - A mutable collection with chainable methods.
  - Underlying data structure is a `list`.
  - Can be converted to `Iter` with the `.iter()` method.
  - Implement MutableSequence Protocol.
- `Dict[K, V]`
  - An immutable mapping with chainable methods.
  - Underlying data structure is a `dict`.
  - Can be converted to `Iter` with the `.iter_items()`, `.iter_keys()`, `.iter_values()` methods.

- `Option[T] | Some[T] | NONE`
  - A type representing an optional value.
  - Provides all methods from the Rust stdlib `Option` Trait (as long as they are applicable/made sense in a Python context).
  - Analog to a superset of `T | None`.
- `Result[T, E] | Ok[T] | Err[E]`
  - A type representing either a success (`Ok[T]`) or failure (`Err[E]`), similar to Rust's `Result` Enum.
  - Provides all methods from the Rust stdlib `Result` Trait (as long as they are applicable/made sense in a Python context).
  - Analog to a superset of `T | Exception`.

### Shared Core Features

All provided classes share the following core methods for enhanced usability:

- **`.from_()`**: Create instances from various inputs. This allow flexible instantiation when needed, whilst keeping the base **init** of the classes without type conversions for performance.

- **`.inspect()`**: Insert functions who compute side-effects in the chain without breaking it (print, mutation of an external variable, logging...). If Option or Result, call the function only if `Some` or `Ok`.

- **`.into()`**:  Provide a `Callable[[Self, P], T]` to convert from **Self** to T in a fluent way.
E.g `Seq[T].into()` can take any function/object that expect a `Sequence[T]` as argument, and return it's result `R`.
Conceptually, replace`f(x, args, kwargs)` with `x.into(f, args, kwargs)`.

## Installation

```bash
uv add pyochain
```

## API Reference 📖

The full API reference can be found at:
<https://outsquarecapital.github.io/pyochain/>

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

### Examples

#### Chained Data Transformations

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

#### Type-Safe Error Handling (`Result` and `Option`)

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

### Philosophy

- **Declarative over Imperative:** Replace explicit `for` and `while` loops with sequences of high-level operations (map, filter, group, join...).
- **Fluent Chaining:** Each method transforms the data and returns a new wrapper instance, allowing for seamless chaining.
- **Lazy and Eager:** `Iter` operates lazily for efficiency on large or infinite sequences, while `Seq` and `Vec` represent materialized sequences for eager operations.
- **Explicit mutability:** `Seq` is the usual return type for most methods who materialize data, hence improving memory efficiency and safety, compared to using list everytime. `Vec` is provided when mutability is required.
- **100% Type-safe:** Extensive use of generics and overloads ensures type safety and improves developer experience.
- **Documentation-first:** Each method is thoroughly documented with clear explanations, and usage examples. Before any commit is made, each docstring is automatically tested to ensure accuracy. This also allows for a convenient experience in IDEs, where developers can easily access documentation with a simple hover of the mouse.
- **Functional and chained paradigm:** Design encourages building complex data transformations by composing simple, reusable functions on known buildings blocks, rather than implementing customs classes each time.

### Inspirations

- **Rust's language and  Rust stdlib:** Emulate naming conventions (`from_()`, `into()`) and leverage concepts from Rust's powerful iterator traits (method chaining, lazy evaluation), Option and Result enums, to bring similar expressiveness to Python.
- **Python iterators libraries:** Libraries like `rolling`, `cytoolz`, and `more-itertools` provided ideas, inspiration, and implementations for many of the iterator methods.
- **PyFunctional:** Although not directly used (because I started writing pyochain before discovering it), also shares similar goals and ideas.

## Key Dependencies and credits

Most of the computations are done with implementations from the `cytoolz`, `more-itertools`, and `rolling` libraries.

An extensive use of the `itertools` stdlib module is also to be noted.

pyochain acts as a unifying API layer over these powerful tools.

<https://github.com/pytoolz/cytoolz>

<https://github.com/more-itertools/more-itertools>

The stubs used for the developpement, made by the maintainer of pyochain, can be found here:

<https://github.com/OutSquareCapital/cytoolz-stubs>
"""

from ._dict import Dict
from ._iter import Iter, Seq, Vec
from ._results import NONE, Err, Ok, Option, Result, ResultUnwrapError, Some

__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "Ok",
    "Option",
    "Result",
    "ResultUnwrapError",
    "Seq",
    "Some",
    "Vec",
]
