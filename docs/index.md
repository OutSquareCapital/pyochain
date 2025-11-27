# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

`pyochain` brings a fluent, declarative API inspired by Rust's `Iterator` and DataFrame libraries like Polars to your everyday Python iterables and dictionaries.

Manipulate data through composable chains of operations, enhancing readability and reducing boilerplate.

## Installation

```bash
uv add pyochain
```

## Overview

### Philosophy

* **Declarative over Imperative:** Replace explicit `for` and `while` loops with sequences of high-level operations (map, filter, group, join...).
* **Fluent Chaining:** Each method transforms the data and returns a new wrapper instance, allowing for seamless chaining.
* **Lazy and Eager:** `Iter` operates lazily for efficiency on large or infinite sequences, while `Seq` represents materialized Sequences for eager operations.
* **100% Type-safe:** Extensive use of generics and overloads ensures type safety and improves developer experience.
* **Documentation-first:** Each method is thoroughly documented with clear explanations, and usage examples.
* **Functional paradigm:** Design encourages building complex data transformations by composing simple, reusable functions.

### Core Components

#### `Iter[T]`

A wrapper for lazy processing of `Iterator` and `Generator` objects.

[See full documentation →](reference/iter.md)

#### `Seq[T]`

A wrapper for eager processing of `Sequence` objects like `list` and `tuple`.

[See full documentation →](reference/seq.md)

#### `Dict[K, V]`

A chainable wrapper for dictionaries.

[See full documentation →](reference/dict.md)

#### `Result[T, E]`

A type for robust error handling, representing either a success (`Ok`) or a failure (`Err`).

[See full documentation →](reference/result.md)

#### `Option[T]`

A type for handling optional values that can be either `Some(value)` or `NONE`.

[See full documentation →](reference/option.md)

## Quick Examples

### Lazy Iteration

```python
import pyochain as pc

# Chain operations on iterables lazily
result = (
    pc.Iter.from_count(1)
    .filter(lambda x: x % 2 != 0)
    .map(lambda x: x ** 2)
    .take(5)
    .into(list)
)
# [1, 9, 25, 49, 81]
```

### Error Handling

```python
import pyochain as pc

def divide(a: int, b: int) -> pc.Result[float, str]:
    if b == 0:
        return pc.Err("Cannot divide by zero")
    return pc.Ok(a / b)

# Safely handle the result
result = divide(10, 0).map_or_else(
    ok=lambda val: f"Success: {val}",
    err=lambda msg: f"Error: {msg}"
)
# "Error: Cannot divide by zero"
```

## Contributing

See [CONTRIBUTING.md](https://github.com/OutSquareCapital/pyochain/blob/master/CONTRIBUTING.md)

## License

MIT License - see [LICENSE.md](https://github.com/OutSquareCapital/pyochain/blob/master/LICENSE.md)
