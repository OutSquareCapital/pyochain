# pyochain ⛓️

**_Fluent method chaining for Python._**

Inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like `Polars`, `pyochain` provide a set of classes with a fluent and declarative API, to work with collections, handle optional values, or manage errors.

## Key Features

- ⛓️ **Declarative & fluent chaining** — Replace `for` loops, None checks, and error handling with chainable methods.
- 🦥 **Lazy-first,  🔒 explicit mutability** — `Iter[T]` for lazy, efficient iterations; `Seq` and `Set` for immutable data; `Vec` and `SetMut` when you need to mutate.
- **Memory efficient** - Almost all methods from `Iter[T]` operate in streaming fashion, and `Vec[T]` provides in-place methods with more memory efficiency than standard list methods (e.g. `x.extend_move(y)` won't create intermediate allocations like `x.extend(y)` followed by `y.clear()`).
- 🎯 **Result and Option types** - Handle `None` and exceptions in a fluent, explicit way.
- 🔥 **Blazingly fast** — Core `Option` and `Result` types are written in Rust for minimal overhead, and iterators use functions from `cytoolz` (Cython) and the stdlib `itertools` for maximum efficiency.
- 🛡️ **100% type-safe** — Full generic support and autocompletion in your IDE.
- 📚 **Accurate Documentation** — Every method is documented and tested with runnable examples.
Every code example in the website (or this README) is also tested, ensuring accuracy and reliability.
- 🔄 **Interoperable** — Seamlessly convert to/from types with various methods like `.into()` and `.collect()`, convert `Iterables` to `Option` or `Result` based on their truthiness, and more.
- 🐍 **Mixins traits** — Extend your own classes with the methods in the mixins provided by the `traits` module.

## Installation

```bash
uv add pyochain # or pip install pyochain
```

[See the package page on Pypi](<https://pypi.org/project/pyochain/>)

## Quick Start

### Iterations

```python
>>> import pyochain as pc
>>> # Lazy processing with Iter
>>> res: pc.Seq[tuple[int, str]] = (
...     pc.Iter.from_count(1)
...     .filter(lambda x: x % 2 != 0)
...     .map(lambda x: x**2)
...     .take(5)
...     .enumerate()
...     .map_star(lambda idx, value: (idx, str(value)))
...     .collect()
... )
>>> res
Seq((0, '1'), (1, '9'), (2, '25'), (3, '49'), (4, '81'))
```

For comparison, the above can be written in pure Python as the following (note that Pylance strict will complain because `itertools.starmap` has not the same overload exhaustiveness as pyochain's `Iter.map_star`):

```python
>>> import itertools
>>>
>>> res: tuple[tuple[int, str], ...] = tuple(
...     itertools.islice(
...         itertools.starmap(
...             lambda idx, val: (idx, str(val)),
...             enumerate(
...                 map(lambda x: x**2, filter(lambda x: x % 2 != 0, itertools.count(1)))
...             ),
...         ),
...         5,
...     )
... )
>>>
>>> res
((0, '1'), (1, '9'), (2, '25'), (3, '49'), (4, '81'))
```

### Result and Option Types

```python

>>> import pyochain as pc
>>> def divide(a: int, b: int) -> pc.Option[float]:
...     return pc.NONE if b == 0 else pc.Some(a / b)
>>> divide(10, 2)
Some(5.0)
>>> divide (10, 0).unwrap_or(-1.0) # Provide a default value
-1.0
>>> # Convert between Collections -> Option -> Result
>>> data = pc.Seq([1, 2, 3])
>>> data.then_some() # Convert Seq to Option
Some(Seq(1, 2, 3))
>>> data.then_some().map(lambda x: x.sum()).ok_or("No values") # Convert Option to Result
Ok(6)
>>> pc.Seq[int](()).then_some().map(lambda x: x.sum()).ok_or("No values")
Err('No values')
>>> pc.Seq[int](()).then_some().map(lambda x: x.sum()).ok_or("No values").ok() # Get the Option back
NONE

```

## Documentation

For comprehensive guides and examples:

[🔍 Types Overview](https://outsquarecapital.github.io/pyochain/core-types-overview/) — Roles, comparisons and visual relationships

[🔄 Interoperability](https://outsquarecapital.github.io/pyochain/interoperability/) - Converting between types

[📖 Examples & Cookbook](https://outsquarecapital.github.io/pyochain/examples/) — Practical patterns and recipes

[📚 Full API Reference](https://outsquarecapital.github.io/pyochain/api-reference/) — Complete API documentation

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
