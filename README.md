# pyochain ⛓️

**_Fluent method chaining for Python._**

Inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like `Polars`, `pyochain` is a no-dependency library providing a set of classes with a fluent API, to work with iterations, collections, handle optional values, or manage errors.

For a quick overview of the core types and their relationships, see the [core types overview](https://outsquarecapital.github.io/pyochain/overview/) page.

## Key Features

- ⛓️ **Declarative & fluent chaining** — Replace `for` loops, None checks, and error handling with chainable methods.
- 🦥 **Lazy-first,  🔒 explicit mutability** — `Iter[T]` for lazy, efficient iterations; `Seq` and `Set` for immutable data; `Vec` and `SetMut` when you need to mutate.
- **Memory efficient** - Almost all methods from `Iter[T]` operate in streaming fashion, and `Vec[T]` provides in-place methods with more memory efficiency than standard list methods (e.g. `x.extend_move(y)` won't create intermediate allocations like `x.extend(y)` followed by `y.clear()`).
- 🎯 **Result and Option types** - Handle `None` and exceptions in a fluent, explicit way.
- 🔥 **Blazingly fast** — Core `Option` and `Result` types are written in Rust for minimal overhead, and iterators use almost always compiled `C` or `Rust` level code, from the `builtins`, `itertools` or custom Pyo3 implementations.
- 🛡️ **100% type-safe** — Full generic support and autocompletion in your IDE.
- 📚 **Accurate Documentation** — Every method is documented and tested with runnable examples.
Every code example in the website (or this README) is also tested, ensuring accuracy and reliability.
- 🔄 **Interoperable** — Seamlessly convert to/from types with various methods like `.into()` and `.collect()`, convert `Iterables` to `Option` or `Result` based on their truthiness, and more.
- 🐍 **Mixins and ABC's** — Extend your own classes with the mixins, Protocol and ABC's provided by the `traits` module.

## Installation

```bash
uv add pyochain # or pip install pyochain
```

[See the package page on Pypi](<https://pypi.org/project/pyochain/>)

## Quick Start

### Iterations

```python
>>> from pyochain import Iter, Seq
>>> # Lazy processing with Iter
>>> res: Seq[tuple[int, str]] = (
...     Iter.from_count(1)
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

>>> from pyochain import Option, NONE, Some, Seq, Vec, Set, Ok, Err, Result
>>> from pyochain.traits import PyoIterable
>>>
>>> def divide(a: int, b: int) -> Option[float]:
...     return NONE if b == 0 else Some(a / b)
>>> divide(10, 2)
Some(5.0)
>>> # Provide a default value
>>> divide(10, 0).unwrap_or(-1.0)
-1.0
>>> # Convert between Collections -> Option -> Result
>>> data = Seq((1, 2, 3))
>>> # Convert Seq to Option
>>> data.then_some()
Some(Seq(1, 2, 3))
>>> # Accept any Pyochain Iterable
>>> def _process(data: PyoIterable[int]) -> str:
...     return data.iter().map(str).join(", ")
>>> # Process only if non-empty, convert Option to Result
>>> data.then(_process).ok_or("No values") 
Ok('1, 2, 3')
>>> Vec(()).then(_process).ok_or("No values")
Err('No values')
>>> # Create empty Set, convert to Result, then back to Option
>>> Set(()).then(_process).ok_or("No values").ok()
NONE
>>> def try_parse_int(s: str) -> Result[int, ValueError]:
...     try:
...         return Ok(int(s))
...     except ValueError as e:
...         return Err(e)
>>> # Type safe exhaustive handling with pattern matching
>>> def handle_result(res: Result[int, ValueError]) -> str:
...     match res:
...         case Ok(value):
...             return f"Parsed value: {value}"
...         case Err(error):
...             return f"Error parsing int!"
>>>
>>> try_parse_int("123").into(handle_result)
'Parsed value: 123'
>>> try_parse_int("abc").into(handle_result)
'Error parsing int!'

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
