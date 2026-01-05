# pyochain ⛓️

**_Fluent method chaining for Python._**

Inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like `Polars`, `pyochain` provide a set of classes with a fluent and declarative API, to work with collections, handle optional values, or manage errors.

## Key Features

- ⛓️ **Declarative & fluent chaining** — Replace `for` loops with chainable methods (`map`, `filter`, `group`, etc.).
- 🦥 **Lazy-first design** — `Iter[T]` for lazy processing; `Seq`, `Vec`, `Set` for materialized collections.
- 🔒 **Explicit mutability** — `Seq` and `Set` for immutable data; `Vec` and `SetMut` when you need to mutate.
- 🛡️ **100% type-safe** — Full generic support and autocompletion in your IDE.
- 📚 **Accurate Documentation** — Every method is documented and tested with runnable examples.
Every code example in the website (or this README) is also tested, ensuring accuracy and reliability.
- ⚡ **Performance-conscious** — Built on `cytoolz` (Cython), `more-itertools`, and stdlib `itertools` for efficiency.
- 🔄 **Interoperable** — Seamlessly convert to/from types with `.into()`, `.collect()`
- 🐍 **Mixins traits** — Extend your own classes with the methods in the mixins provided by the `traits` module.

## Quick Start

```python
>>> import pyochain as pc
>>> # Lazy processing with Iter
>>> res: pc.Seq[int] = (
...     pc.Iter.from_count(1)
...     .filter(lambda x: x % 2 != 0)
...     .map(lambda x: x**2)
...     .take(5)
...     .collect()
... )
>>> res
Seq(1, 9, 25, 49, 81)
>>> # Type-safe error handling with Result
>>> def divide(a: int, b: int) -> pc.Result[float, str]:
...     return pc.Err("Division by zero") if b == 0 else pc.Ok(a / b)
>>> divide(10, 0).unwrap_or(0.0)
0.0
```

## Installation

```bash
uv add pyochain # or pip install pyochain
```

[See the package page on Pypi](<https://pypi.org/project/pyochain/>)

## Documentation

For comprehensive guides and examples:

- **Why?** → [📘 User Guide](https://outsquarecapital.github.io/pyochain/user-guide/) — Concepts and mental models
- **With what?** → [🔍 Core Types Overview](https://outsquarecapital.github.io/pyochain/core-types-overview/) — Type overview, comparisons and visual relationships
- **How?** → [📚 Full API Reference](https://outsquarecapital.github.io/pyochain/api-reference/) — Complete API documentation
- **For when?** → [📖 Examples & Cookbook](https://outsquarecapital.github.io/pyochain/examples/) — Practical patterns and recipes

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

## Contributing

Want to contribute? Read our [contributing guide](CONTRIBUTING.md)

## Key Dependencies and credits

Most of the computations are done with implementations from, itertools, `cytoolz` and `more-itertools`.

pyochain acts as a unifying API layer over these powerful tools.

<https://github.com/pytoolz/cytoolz>

<https://github.com/more-itertools/more-itertools>

The stubs used for the development, made by the maintainer of pyochain, can be found here:

<https://github.com/OutSquareCapital/cytoolz-stubs>
