# User Guide

`pyochain` provides a small set of core types (`Iter`, `Seq`, `Vec`, `Set`, `SetMut`, `Dict`, `Option`, `Result`) to write Python code as data pipelines:

- build transformations as a chain (instead of step-by-step loops)
- keep **data flow** and **side effects** separated
- make “may be missing” (`Option`) and “may fail” (`Result`) explicit

This guide focuses on concepts and a mental model. For exhaustive examples and method lists, prefer:

- the reference pages in `docs/reference/`
- the cookbook in `docs/examples.md`

---

## Data types and structures

Pyochain types are small wrappers over familiar Python concepts. Pick them based on two dimensions:

- evaluation: lazy (streaming) vs eager (materialized)
- mutability: immutable vs mutable

### `Iter[T]`: streaming and lazy

`Iter[T]` wraps a Python `Iterator[T]` and is designed for pipelines.

- transformation methods (`map`, `filter`, `take`, …) are lazy
- terminal methods (`collect`, `sum`, `for_each`, …) consume the iterator

Important: `Iter` is single-use. Once consumed, it is exhausted.

### `Seq[T]`: eager and immutable

`Seq[T]` is tuple-backed: it is materialized, reusable, and great for “hold this in memory and traverse multiple times”.

If you want laziness again, call `.iter()`.

### `Vec[T]`: eager and mutable

`Vec[T]` is list-backed: use it when you need in-place mutations (append/extend/insert, etc.).

### `Set[T]` and `SetMut[T]`: uniqueness

Use `Set[T]` (immutable) or `SetMut[T]` (mutable) when you need uniqueness and fast membership tests.

### `Dict[K, V]`: key/value transformations

`Dict[K, V]` is a mapping wrapper with a fluent API.

- for typed lookup, prefer `get_item(key) -> Option[V]` over `dict.get(key) -> V | None`

### Which one should I use?

| You need… | Use |
| --- | --- |
| a streaming pipeline, laziness by default | `Iter[T]` |
| an immutable materialized collection | `Seq[T]` |
| a mutable materialized collection | `Vec[T]` |
| uniqueness / membership tests | `Set[T]` / `SetMut[T]` |
| key/value operations with `Option` lookup | `Dict[K, V]` |

For a compact overview, see `docs/core-types-overview.md`.

Reference pages:

- [`Iter[T]`](reference/iter.md)
- [`Seq[T]`](reference/seq.md)
- [`Vec[T]`](reference/vec.md)
- [`Set[T]`](reference/set.md)
- [`SetMut[T]`](reference/setmut.md)
- [`Dict[K, V]`](reference/dict.md)
- [`Option[T]`](reference/option.md)
- [`Result[T, E]`](reference/result.md)

---

## Functions as values (callables)

Most transformation methods take a callable (function/lambda). Treat these callables as reusable “blocks” you can compose.

```python
>>> import pyochain as pc
>>> def is_even(n: int) -> bool:
...     return n % 2 == 0
>>> def square(n: int) -> int:
...     return n * n
>>> out: pc.Seq[int] = pc.Iter(range(10)).filter(is_even).map(square).collect()
>>> out
Seq(0, 4, 16, 36, 64)
```

Guidelines:

- use small named `def` when reused, or when type checkers struggle with lambdas
- keep lambdas for tiny one-liners
- for tuple-like elements, prefer `map_star(func)` to unpack into arguments

Terminal operations are where an `Iter[T]` is actually consumed (e.g. `collect`, `sum`, `for_each`).

---

## Lazy API: when to use which

Pyochain is “lazy-first” in one very specific sense: `Iter` chains are not executed until you consume them.

There is no query planner: pyochain will not rewrite or optimize your pipeline. The main benefit of laziness here is simply “do no work until you must”.

### Prefer lazy when

- you want to stream values and avoid materializing intermediate collections
- you want to combine transformations before producing a final result
- you only need one pass over the data

### Prefer eager when

- you need to inspect intermediate results (exploration/debugging)
- you need multiple passes (e.g. “filter, then later reuse the filtered data twice”)
- you need random access (indexing), stable size, or repeated iteration

Practical pattern: sample a pipeline without consuming everything.

```python
>>> import pyochain as pc
>>> head: pc.Seq[int] = pc.Iter(range(1_000_000)).map(lambda x: x * 2).take(5).collect()
>>> head
Seq(0, 2, 4, 6, 8)
```

---

## Side effects: `inspect` vs `for_each`

Side effects (logging, printing, metrics) are often necessary, but they shouldn’t destroy composability.

Use:

- `inspect(...)` to observe intermediate values while keeping the chain intact
- `for_each(...)` when the pipeline’s purpose is the side effect (terminal, returns `None`)

```python
>>> import pyochain as pc
>>> _ = pc.Iter(range(5)).inspect(lambda _: print("about to consume")).map(lambda n: n + 1).for_each(print)
about to consume
1
2
3
4
5
```

---

## Missingness and failure are data: `Option` and `Result`

Pyochain’s “types for control flow” are meant to keep “expected absence” and “recoverable failure” explicit.

### `Option[T]`: expected absence

Use `Option[T]` when “no value” is a normal outcome (lookups, searches, optional config).

```python
>>> import pyochain as pc
>>> cfg = pc.Dict.from_kwargs(host="localhost")
>>> port: int = cfg.get_item("port").map(int).unwrap_or(3000)
>>> port
3000
```

### `Result[T, E]`: recoverable failure

Use `Result[T, E]` when something may fail and the caller should decide what to do.

A pragmatic Python pattern is “exceptions at the boundary”: convert raised exceptions at IO/parsing boundaries into a `Result`, then keep the rest of your pipeline exception-free.

```python
>>> import pyochain as pc
>>> def parse_port(value: str) -> pc.Result[int, str]:
...     try:
...         port = int(value)
...     except ValueError:
...         return pc.Err("port must be an integer")
...     if 1 <= port <= 65535:
...         return pc.Ok(port)
...     return pc.Err("port out of range")
>>> port: int = parse_port("8080").unwrap_or(3000)
>>> port
8080
```

Decision rule:

- `Option[T]`: missing is expected
- `Result[T, E]`: failure is expected and recoverable
- exceptions: bugs / invariant violations

---

## Interoperability: `.into(...)` and `.collect(...)`

Two conversions matter most:

- `.collect(...)`: materialize an `Iter` into a pyochain collection (`Seq` by default)
- `.into(func_or_type)`: convert by calling a function/type with the current object

```python
>>> import pyochain as pc
>>> as_seq: pc.Seq[int] = pc.Iter(range(5)).collect()
>>> as_seq
Seq(0, 1, 2, 3, 4)
>>> as_list: list[int] = as_seq.into(list)
>>> as_list
[0, 1, 2, 3, 4]
```

This makes gradual adoption practical: internal logic can use pyochain types, while boundaries can convert to/from native Python types.

---

## Where to go next

- [`docs/examples.md`](examples.md)
- [`docs/core-types-overview.md`](core-types-overview.md)
