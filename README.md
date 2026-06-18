# pyochain ⛓️

**_Fluent method chaining for Python._**

Inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like `Polars`, `pyochain` is a no-dependency library providing a set of classes with a fluent API, to work with iterations, collections, handle optional values, or manage errors.

## Key Features

### ⛓️ Declarative & fluent chaining

Replace `for` loops, None checks, and error handling with chainable methods.

### 🎯 Result and Option types

Handle `None` and exceptions in a fluent, explicit way.

### 🔥 Blazingly fast

Core `Option` and `Result` types are implemented in Rust for minimal overhead, and iterators use almost always compiled `C` or `Rust` level code, from the python builtins, `itertools`, or custom Pyo3 implementations.

### 🛡️ 100% type-safe

IDE autocompletion is a primary concern, and pyochain bring exhaustive overloads and generics support for all it's constructs.

### 📚 Accurate, tested Documentation

Every method is documented and tested with runnable examples.

Every code example in the website (or this README) is also tested, ensuring accuracy and reliability.

### 🔄 Interoperable

Seamlessly convert to/from types with various methods like `.pipe()` and `.collect()`, convert `Iterables` to `Option` or `Result` based on their truthiness, and more.

### 🐍 Mixins and ABC's

Extend your own classes with the mixins, Protocol and ABC's provided by the `abc` module.

---

## Installation

```bash
uv add pyochain # or pip install pyochain
```

[See the package page on Pypi](<https://pypi.org/project/pyochain/>)

---

## Quick Start

### Iterations

Below is an example of how to use `Iter`, and how it compares to a pure Python implementation using `itertools`:

```python
from pyochain import Iter, Seq
import itertools

wanted = ((0, "1"), (1, "9"), (2, "25"), (3, "49"), (4, "81"))

pyochain_res = (
    Iter
    .from_count(1)
    .filter(lambda x: x % 2 != 0)
    .map(lambda x: x**2)
    .take(5)
    .enumerate()
    .map_star(lambda idx, value: (idx, str(value)))
    .collect(tuple)
)
py_res = tuple(
    itertools.islice(
        itertools.starmap(
            lambda idx, val: (idx, str(val)),
            enumerate(
                map(lambda x: x**2, filter(lambda x: x % 2 != 0, itertools.count(1)))
            ),
        ),
        5,
    )
)
assert pyochain_res == py_res == wanted
```

### Result and Option Types

```python
from pyochain import Option, NONE, Some, Seq, Vec, Set, Ok, Err, Result
from pyochain.abc import PyoIterable


def divide(a: int, b: int) -> Option[float]:
    return NONE if b == 0 else Some(a / b)


assert divide(10, 2) == Some(5.0)
# Provide a default value
assert divide(10, 0).unwrap_or(-1.0) == -1.0
# Convert between Collections -> Option -> Result
tup = (1, 2, 3)
seq = Seq(tup)
assert seq.then_some() == Some(Seq(tup))
assert seq.then_some().ok_or("No values").unwrap() == Seq(tup)


# Accept any Pyochain Iterable
def _process(data: PyoIterable[int]) -> str:
    return data.iter().map(str).join(", ")


# Process only if non-empty, convert Option to Result
assert seq.then(_process).ok_or("No values").unwrap() == "1, 2, 3"
assert (
    Vec(()).then(_process).ok_or("No values").expect_err("expected error")
    == "No values"
)
# Create empty Set, convert to Result, then back to Option
assert Set(()).then(_process).ok_or("No values").ok() is NONE
```

#### Type safe exhaustive handling with pattern matching

Type checkers will ensure that all cases are handled when matching on `Option` and `Result` types.

```python
from pyochain import Result, Ok, Err


def try_parse_int(s: str) -> Result[int, ValueError]:
    try:
        return Ok(int(s))
    except ValueError as e:
        return Err(e)


def handle_result(res: Result[int, ValueError]) -> str:
    match res:
        case Ok(value):
            return f"Parsed value: {value}"
        case Err(_):
            return f"Error parsing int!"


assert try_parse_int("123").pipe(handle_result) == "Parsed value: 123"
assert try_parse_int("abc").pipe(handle_result) == "Error parsing int!"
```

## Documentation

For comprehensive guides and examples:

[🔍 Types Overview](https://outsquarecapital.github.io/pyochain/overview/) — Roles, comparisons and visual relationships

[📚 Full API Reference](https://outsquarecapital.github.io/pyochain/api-reference/) — Complete API documentation

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

## Contributing

Want to contribute? Read our [contributing guide](CONTRIBUTING.md)

## Credits

Most of the custom computation algorithms have been inspired by implementations from itertools, [`cytoolz`](https://github.com/pytoolz/cytoolz) and [`more-itertools`](https://github.com/more-itertools/more-itertools).
