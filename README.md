# pyochain ⛓️

**_Fluent method chaining for Python._**

## Summary

Inspired by Rust's `Iterator`, `Result`, `Option`, and DataFrame libraries like `Polars`, `pyochain` provide a set of classes with a fluent,declarative API, to work with collections, handle optional values, or manage errors.

## Quick Start

Install the latest release from **PyPI**:

```bash
uv add pyochain # or pip install pyochain
```

Import the library and start chaining methods:

```python
import pyochain as pc

# Lazy processing with Iter
res: pc.Seq[int] = (
    pc.Iter.from_count(1)
    .filter(lambda x: x % 2 != 0)
    .map(lambda x: x**2)
    .take(5)
    .collect()
)
# → Seq(1, 9, 25, 49, 81)


# Type-safe error handling with Result
def divide(a: int, b: int) -> pc.Result[float, str]:
    return pc.Err("Division by zero") if b == 0 else pc.Ok(a / b)


divide(10, 0).unwrap_or(0.0)  # → 0.0

```

### Philosophy

- **Declarative over Imperative:** Replace explicit `for` and `while` loops with sequences of high-level operations (map, filter, group, join...).
- **Fluent Chaining:** Most methods transforms the data and returns a new wrapper instance, allowing for seamless chaining.
- **Lazy first:** All methods on collections that use an Iterator (think most for loop) and do not need to materialize data immediately are in `Iter[T]`.
Only methods that directly returns booleans or single values are shared between the `Iterable` classes (`Iter`, `Seq`, `Vec`, `Set`, `SetMut`) via their common base class.
This encourages the use of lazy processing by default (since you have to explicitly call `iter()` to get access to most methods), and collecting only at the last possible moment.
- **Explicit mutability:** `Seq` is the usual return type for most methods who materialize data, hence improving memory efficiency and safety, compared to using list everytime. `Vec` is provided when mutability is required. Same for `Set` and `SetMut`. In python, set is the "default" set type (constructor and shorter name), but in pyochain Set is a frozenset.
- **100% Type-safe:** Extensive use of generics and overloads ensures type safety and improves developer experience. The library is fully typed and autocompletion is a central concern.
- **Documentation-first:** Each method is thoroughly documented with clear explanations, and usage examples. Before any commit is made, each docstring is automatically tested to ensure accuracy. This also allows for a convenient experience in IDEs, where developers can easily access documentation with a simple hover of the mouse, with a guarantee that the examples work as intended.
- **Functional and chained paradigm:** Design encourages building complex data transformations by composing simple, reusable functions on known buildings blocks, rather than implementing customs classes each time.
- **Interoperability:** pyochain types can be created from and converted back to their Python equivalents seamlessly, allowing for easy integration into existing codebases.
- **Performance-conscious:** While prioritizing readability and maintainability, pyochain leverage the cytoolz (`Cython`), more-itertools, and stdlib itertools (`C`) libraries for efficient implementations of core algorithms.
- **User-friendly API**: Method names and behaviors are designed to be intuitive and consistent with Python and Rust standard libraries, reducing the learning curve for new users. pyochain also provides improvements over standard library functions where applicable, such as `Namedtuples` for `Iter.enumerate()`, `Dict.iter()` (dict.items()), `Iter.group_by()` (over itertools.groupby), etc.

## Documentation

For comprehensive guides and examples:

- **[Examples & Cookbook](https://outsquarecapital.github.io/pyochain/examples/)** — Practical use cases and code examples
- **[Core Types Overview](https://outsquarecapital.github.io/pyochain/core-types-overview/)** — Detailed type explanations and comparisons
- **[Full API Reference](https://outsquarecapital.github.io/pyochain/)** — Complete API documentation

## Notice on Stability ⚠️

`pyochain` is currently in early development (< 1.0), and the API may undergo significant changes multiple times before reaching a stable 1.0 release.

## Contributing

Want to contribute? Read our [contributing guide](CONTRIBUTING.md)

## Key Dependencies and credits

Most of the computations are done with implementations from the `cytoolz` and `more-itertools` libraries.

An extensive use of the `itertools` stdlib module is also to be noted.

pyochain acts as a unifying API layer over these powerful tools.

<https://github.com/pytoolz/cytoolz>

<https://github.com/more-itertools/more-itertools>

The stubs used for the developpement, made by the maintainer of pyochain, can be found here:

<https://github.com/OutSquareCapital/cytoolz-stubs>
