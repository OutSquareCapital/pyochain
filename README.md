# pyochain

`pyochain` is a python library that provides various classes with a fluent API, to work with iterations, collections, handle optional values, manage errors, and more.

The API and functionalities are inspired by Rust's `Iterator`, `Result`, `Option`, and libraries like `Polars`, `toolz` and `more-itertools`.

## Key Features

- `Option[T]` to handle optional values instead of `T | None`.
- `Result[T, E]` to handle success and error paths instead of `try.. except` blocks.
- `Iterator` types covering all python builtins and `itertools`, as well as many methods from Rust's `Iterator` and `more-itertools`.
- `Collection` types covering all built-in python types, and additional ones like no-copy slice views.
- `ABC`'s hierarchy for duck typing, shared methods, and the possibility to implement your own subclasses.
- `Mixin`'s addable to any class, to provide a fluent API , or `Option`/`Result` conversions on truthiness evaluation.
- **Compiled** in Rust for maximum performance with Pyo3.
- **Fluent API** design for chaining method calls, to read your code just like a book => from top to bottom, left to right.
- **First class static typing support**: Generics, overloads, and pattern matching for `Option` and `Result` types.

That's why it's called `pyochain`: it allows you to build *chains* of operations on your data, with code compiled in Rust thanks to [*Pyo3*](https://pyo3.rs/).

---

## Installation

```bash
uv add pyochain # or pip install pyochain
```

## Links

[🐍Pypi](https://pypi.org/project/pyochain/)

[📚 Full API Reference](https://outsquarecapital.github.io/pyochain/api-reference/)

[📄 README.md](https://github.com/OutSquareCapital/pyochain)

[🚀 Getting started](https://github.com/OutSquareCapital/pyochain#getting-started)

## Why use pyochain?

### 🔥 Blazingly fast

Being statically compiled, pyochain is by design order of magnitude faster than other similar python libraries.

For example, a simple, single object creation like `x = Vec([1, 2, 3])` take 30% less time than if `Vec` was implemented in pure Python.

This speed-up is only exacerbated for `Iterator` methods and classes, often up to **2x to 10x** faster than `more-itertools` equivalents.

Even when the source code was still mostly python, great care had been taken to optimize performance, which is probably why it was already ranked as the fastest library in its category in [this comparison](https://www.reddit.com/r/Python/comments/1rj3ct7/a_comparison_of_rustlike_fluent_iterator_libraries/) (at this point, only `Result` and `Option` were compiled, which were not relevant for this benchmark).

### 🌍 Rich ecosystem

Pyochain provides a wide range of features designed to interact with each other and the wider Python ecosystem seamlessly, to act as a drop-in replacement for many built-in types and functions.

There's many additional functionalities planned for the future, including `Array`, sorted collections, unique mutable sequences, and more.

### 🛡️ 100% type-safe

IDE autocompletion is a primary concern, and pyochain brings exhaustive overloads and generics support for all its constructs.

It's even more complete than typeshed in certain cases, for example with `map_star` fully typed regarding arguments and return types, while `itertools.starmap` is not.

This is the hardest part to test when developing the library, so if you encounter any typing issues, please report them!

### 📚 Accurate, tested Documentation

Every method and type is exhaustively documented, and contains runnable examples that are tested for correctness.

Even this README is tested!

---

## Getting started

### Mixin's

`Pipe`, `Tap`, `Checkable`, and other mixins provide a fluent API for any class, allowing to chain method calls.

They don't depend on internal state except `__bool__`, thus making them universally applicable to any subclass, including your own.

Users of `pandas`, `polars` or the Rust crate `tap` will feel right at home with `pipe()`, while Rust developers will appreciate the `Checkable` type, providing methods like `then`, `then_some`, or `ok_or_else`, evaluating the instance truthiness to return corresponding `Option` or `Result` types, just like the `bool` methods in Rust.

All pyochain types herit from them, making control flow for collection emptyness or error handling a natural part of a pipeline.

### Iterators

Pyochain provide various `Iterator` types with a fluent API.

Their methods bring functionnalities from various sources:

- Python `builtins` (fully covered) **=>** `zip`, `map`, etc...
- `itertools` module (fully covered) **=>** `chain`, `combinations`, etc...
- Rust `std::iter::Iterator` **=>** `try_collect`, `partition`, etc...
- `more-itertools` library **=>** `all_unique`, `arg_max`, `tail`, etc...
- `toolz`  library **=>** `map_juxt`, `count`, `first`, etc...

They can be used to build complex pipelines of transformations, filters, and aggregations in a readable way (no nested loops or comprehensions), without creating intermediate collections (lazy execution), thus improving performance and memory usage.

Many methods act just like their Rust counterparts: `filter_map` filter the `Option` returned by a closure, `find` return the first element matching a predicate wrapped in an `Option`, etc...

All the source code related to iterators is implemented in Rust and call either in-house implementations or CPython builtins.

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

### Collections

Each python built-in collection type (list, tuple, range, dict, set, etc...) has a corresponding pyochain type, with  additional collections like `SliceView` (no copy view of a slice), or `StableSet` (a mutable set that preserves insertion order), and more planned for the future.

Many methods are designed to interoperate with the rest of the library: `Dict::get_item` or `PyoIterator::next` return an `Option`, `PyoIterator::try_fold` a `Result`, `Vec::drain` a `PyoIterator`, and so on.

```python
from pyochain import Dict, Iter, Some
from pyochain.collections import StableSet

names = ["Charlie", "Alice", "Bob", "Alice"]


# Create a Dict from an iterator of key-value pairs
data = Iter.from_count().zip(names).collect(Dict)
assert data == Dict({0: "Charlie", 1: "Alice", 2: "Bob", 3: "Alice"})


# try_insert returns a Result, which is Err if the key already exists
err = data.try_insert(1, "David")
assert err.is_err()

# sort return a Vec
vals = data.values().iter().map(str.upper).sort()

assert vals.first() == "ALICE"
assert vals.len() == 4
# Modify the Vec in place with retain according to the predicate
vals.retain(lambda x: x.endswith("E"))
assert vals.len() == 3

# Create a set of unique names, preserving insertion order, with StableSet
unique_names = vals.pipe(StableSet)
assert unique_names == {"ALICE", "CHARLIE"}
assert unique_names.iter().next() == Some("ALICE")
```

### Result and Option

Handle `None` and exceptions in an explicit way with dedicated types, instead of relying on implicit truthiness checks or try/except blocks.

Success and failure can respectively be represented by `Ok[T]` and `Err[E]` types, while optional values can be represented by `Some[T]` and `Null`.

This make each path explicit, less error-prone, and more readable, by replacing nested `try/except` blocks and `if x is not None` checks by a single pipeline like `x.map().and_then().unwrap_or()`.

For users familiar with this pattern, almost all methods present in their Rust counterparts are available, as well as additional convenience methods for broad python ecosystem compatibility, like `unwrap_or_none()` (heresy, I know).

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

### ABC's

A class hierarchy mimicking the `collections.abc` module is provided, each only requiring the same dunders as it's standard library counterpart (e.g `__iter__` for `PyoIterable`), while providing many rust-inspired additional methods, like `PyoMutableSequence::retain`, `PyoMutableMapping::try_insert`, and of course the numerous functionnalities from `PyoIterator`.

All concretes `Iterator` and `Collection` types implement them, so you can use them for type checking, implement your own subclasses, and seamlessly replace python ecosystem types, without trade-offs.

```python
from pyochain import Some
from pyochain.abc import PyoSequence, PyoIterable
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(slots=True)
class MySequence(PyoSequence[int]):
    data: list[int]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> int:
        return self.data[index]


x = MySequence([1, 2, 3])
# Call any method from PyoSequence, like first(), last(), get(), etc...
assert x.get(2) == Some(3)
# Convert to an Iterator and immediately benefit from all the Iterator methods
assert x.iter().map(lambda x: x**2).collect(tuple) == (1, 4, 9)
# Convert to an Option just like pyochain core API
assert x.then_some() == Some(x)
# Works with runtime instance and subclass type checking
assert isinstance(x, PyoSequence)
assert isinstance(x, PyoIterable)
assert isinstance(x, Sequence)
assert issubclass(MySequence, PyoSequence)
assert issubclass(MySequence, Sequence)
```

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

## Contributing

We are actively looking for contributors to help us improve `pyochain`! If you are interested in contributing, please read our [contributing guide](CONTRIBUTING.md) for more information on how to get started.

## Credits

Most of the custom computation algorithms have been inspired by implementations from itertools, [`cytoolz`](https://github.com/pytoolz/cytoolz) and [`more-itertools`](https://github.com/more-itertools/more-itertools).

[Pyo3](https://pyo3.rs/) is used to compile the library in Rust, and provide a seamless integration with Python.

[Polars](https://www.pola.rs/) has been what made me realize that reading my code from top to bottom was a better way to write python code, and also introduced me to Rust.

## Star History

<a href="https://www.star-history.com/?repos=outsquarecapital%2Fpyochain&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=outsquarecapital/pyochain&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=outsquarecapital/pyochain&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=outsquarecapital/pyochain&type=date&legend=top-left" />
 </picture>
</a>
