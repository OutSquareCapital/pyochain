# User Guide

`pyochain` is a small set of core types (`Iter`, `Seq`, `Vec`, `Set`, `Dict`, `Option`, `Result`) designed to make Python code more declarative:

- build pipelines of transformations instead of step-by-step loops
- separate **data flow** from **side effects**
- make “may fail” and “may be missing” explicit in the type system

This guide focuses on the mental model and the “why”. For exhaustive examples and method lists, prefer the reference pages (see `docs/reference/`) and the cookbook (`docs/examples.md`).

---

## A Mental Model: Adapters vs Terminal Operations

Pyochain’s API is strongly inspired by Rust’s iterator style:

- **adapter methods** transform a container into another container (still chainable)
- **terminal methods** consume the pipeline and produce a final value

For `Iter[T]`, adapter methods are lazy (they build a *description* of work). Nothing is executed until a terminal method runs.

Common adapters:

- `map`, `filter`, `filter_map`, `take`, `skip`, `flat_map`, `flatten`, `sort` (note: `sort` materializes into a `Vec`)

Common terminals:

- `collect` (materialize into `Seq` by default)
- `length`, `sum`, `min`, `max`, `fold`, `reduce`
- `for_each` (side effects)

Minimal example (adapter chain + terminal):

```python
import pyochain as pc

out: pc.Seq[int] = (
    pc.Iter(range(10)).filter(lambda n: n % 2 == 0).map(lambda n: n * n).collect()
)
```

### `Iter` is single-use

`Iter[T]` wraps a Python `Iterator[T]`. Like any iterator, it is exhausted after consumption.

- If you need to traverse data multiple times, materialize once with `.collect()` to a `Seq`/`Vec`/`Set`.
- If you already have `Seq`/`Vec`/`Set`/`Dict`, you can switch to lazy mode with `.iter()`.

---

## Picking the Right Container

You don’t need to memorize every type; think in terms of *evaluation* and *mutability*:

| You need… | Use |
| --- | --- |
| streaming / laziness / “pipeline first” | `Iter[T]` |
| materialized, reusable, immutable data | `Seq[T]` |
| materialized and mutable data | `Vec[T]` |
| uniqueness / membership tests | `Set[T]` or `SetMut[T]` |
| key/value transformations with a fluent API | `Dict[K, V]` |

Notes:

- `Seq[T]` is tuple-backed. It supports indexing, but pyochain also provides `nth(index)` as a terminal method.
- `Vec[T]` is list-backed (`MutableSequence`); you can use list-like methods such as `append`, `extend`, `insert`, etc.
- `Dict[K, V]` implements `MutableMapping` and also offers `get_item(key) -> Option[V]` for typed lookup.

For a compact overview table, see `docs/core-types-overview.md`.

---

## Functions as Values (Closures in Python)

Most pyochain methods take callables (functions/lambdas). This is the “functional” part: you describe transformations by passing behavior.

Guidelines that tend to scale well:

- Prefer small named functions (`def`) when the logic is reused or when type checkers struggle with complex lambdas.
- Use lambdas for tiny one-liners.
- When your pipeline elements are “tuple-like”, prefer `map_star(func)` to unpack items into function arguments.

Example with named predicates and mappers:

```python
import pyochain as pc


def is_even(n: int) -> bool:
    return n % 2 == 0


def square(n: int) -> int:
    return n * n


out: pc.Seq[int] = pc.Iter(range(10)).filter(is_even).map(square).collect()
```

---

## Side Effects: `inspect` vs `for_each`

Side effects are often necessary (logging, metrics, printing), but they shouldn’t destroy composability.

Use:

- `inspect(...)` when you want to **observe** intermediate values without breaking the chain
- `for_each(...)` when the entire purpose of the pipeline is the side effect (and you want to consume it)

Key difference:

- `inspect` returns the original container for chaining
- `for_each` is terminal and returns `None`

```python
import pyochain as pc

pc.Iter(range(5)).inspect(lambda it: print("about to consume")).map(lambda n: n + 1).for_each(print)
```

---

## Error Handling: `Result[T, E]` (Recoverable Errors)

Like Rust, pyochain encourages distinguishing:

- **recoverable** errors: you expect them to happen sometimes (bad user input, missing file, timeouts)
- **unrecoverable** errors: bugs or invariant violations (these can still raise)

In pyochain, recoverable errors are represented by `Result[T, E]`.

### Why this helps

Compared to exceptions as control flow:

- the *type signature* communicates “this may fail”
- failures are chainable with `map`, `and_then`, `or_else`, etc.
- you can build linear pipelines without nested `try/except`

### Practical pattern: “exceptions at the boundary”

Python libraries often raise exceptions. A good compromise is:

1) catch exceptions at IO / parsing boundaries
2) convert to `Result`
3) keep the rest of the pipeline exception-free

```python
import pyochain as pc


def parse_port(value: str) -> pc.Result[int, str]:
    try:
        port = int(value)
    except ValueError:
        return pc.Err("port must be an integer")

    if 1 <= port <= 65535:
        return pc.Ok(port)
    return pc.Err("port out of range")


port: int = parse_port("8080").unwrap_or(3000)
```

### Reading the combinators

You can mentally model the core methods like this:

- `map(f)`: apply `f` to the `Ok` value
- `map_err(f)`: apply `f` to the `Err` value
- `and_then(f)`: sequence fallible operations (a “flat_map” for `Result`)
- `unwrap_or(default)`: extract the value with a fallback
- `map_or_else(ok=..., err=...)`: produce a final value from either branch

---

## Optional Values: `Option[T]` (Expected Absence)

`Option[T]` is for “missingness”, not for “failure”. Use it when “no value” is an expected outcome.

Typical sources:

- optional config keys
- lookups in mappings
- search operations

### Prefer typed lookups with `Dict.get_item`

Instead of `dict.get(...) -> T | None`, you can use `Dict.get_item(...) -> Option[T]` and stay in the fluent world:

```python
import pyochain as pc

cfg = pc.Dict.from_kwargs(host="localhost")
port: int = cfg.get_item("port").map(int).unwrap_or(3000)
```

### `Option` combinators mirror `Result`

- `map(f)`: apply `f` to a `Some` value
- `and_then(f)`: chain optional operations
- `unwrap_or(default)`: extract with fallback
- `ok_or(err)`: convert `Option[T]` to `Result[T, E]`

---

## `Option` vs `Result` vs Exceptions

Use this as a default decision rule:

- `Option[T]`: the value may be missing and that’s *expected*
- `Result[T, E]`: the operation may fail and the caller should decide what to do
- exceptions: bugs or invariants (the program should not continue normally)

---

## Interoperability: `.into(...)` and `.collect(...)`

Two concepts matter:

- `.collect(...)` materializes an `Iter` into a pyochain collection (`Seq` by default)
- `.into(func_or_type)` converts to *anything* by calling a function with the current object

Examples:

```python
import pyochain as pc

as_seq: pc.Seq[int] = pc.Iter(range(5)).collect()
as_list: list[int] = as_seq.into(list)
```

This makes gradual adoption practical: you can keep using external libraries that expect native Python types while writing your internal logic with pyochain types.

---

## Note on Polars-style “contexts” (`with_columns`)

Polars expressions are *descriptions* of computations that only materialize inside a context (`select`, `with_columns`, `filter`, …). That’s conceptually similar to pyochain adapters vs terminals: build descriptions first, execute later.

When integrating pyochain with Polars, one pragmatic use case is to generate repeated expressions (e.g., a family of derived columns) without manual loops:

```python
import polars as pl

import pyochain as pc

out: pl.LazyFrame = (
    pc.Seq([1, 2, 3])
    .into(lambda x: pl.LazyFrame({"x": x}))
    .with_columns(
        pc.Iter(["a", "b", "c"]).map(lambda name: pl.col("x").mul(2).alias(f"{name}_x"))
    )
)

```

This can be more concise and coherent than mixing polars fluent API calls with Python loops.
You also get the benefit of pyochain’s rich combinators for transformations, and lazy evaluation.
This means that expressions AND computations are lazy.
