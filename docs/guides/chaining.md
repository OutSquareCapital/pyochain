# Chaining: Passing Values Between Functions

`pyochain` is built around the idea of **staying in a fluent chain** while still being able to call your own functions, observe what happens, and eventually get back to plain Python values.

This guide explains how data flows between:

- the wrappers (`Iter`, `Seq`, `Dict`, `Result`, `Option`, …),
- your own functions (pure or not),
- and the outside world (final values, side effects).

We will mostly classify methods by **behaviour**, not by concrete type.

## Behavioural categories

- `pipe` / `tap`: work on the *wrapper* itself.
- `map` on `Iter` / `Seq` / `Option` / `Result` vs observation methods (`inspect`, `inspect_err`, `for_each`, `peek`).
- `from_`, `apply`, `into`: bridge between plain Python values and `pyochain` wrappers.

Along the way, we also highlight **which wrappers expose what**, so you can quickly see what is available on `Iter` vs `Seq` vs `Result`, and how they fit together.

---

## 1. Working at the Wrapper Level: `pipe` and `tap`

At the highest level, you sometimes want to pass the **wrapper itself** to a function. In other words, your function wants to receive `pc.Seq[int]` or `pc.Iter[str]`, not a raw `list[int]` or `Iterator[str]`.

This is where `pipe` and `tap` live. Both are defined on the shared `Pipeable` base class, so they are available on **all** `pyochain` wrappers: `Iter`, `Seq`, `Dict`, `Option`, `Result`, etc.

### 1.1 `pipe`: delegate a sub-pipeline

`pipe` passes the current wrapper instance as the first argument to your function and returns **whatever the function returns**.

Conceptually, `pipe` lets you *name* a part of your chain:

```python
def maybe_sum(xs: pc.Seq[int]) -> pc.Option[int]:
    if xs.count() == 0:
        return pc.NONE
    return pc.Some(xs.sum())

pc.Seq(range(5)).pipe(maybe_sum)  # Seq[int] -> Option[int]
```

You can think of `pipe` as "call this helper that knows about the wrapper type":

- Use `pipe` when the helper function **works in terms of wrappers** (`pc.Seq`, `pc.Iter`, `pc.Result`, …).
- Use it to implement branching logic (e.g. return `Option` from `Seq`, or turn a `Result` into another `Result`).

### 1.2 `tap`: side effects on the wrapper

`tap` also receives the wrapper, but it always returns the **same instance**, unchanged.

The intent of `tap` is: *"run some side-effect, but don’t affect the logical value of the chain"*.

```python
pc.Seq([1, 2, 3]).tap(print).last() # logs the whole Seq
```

Typical use cases:

- Debugging / logging the state of the wrapper at some point in the chain.
- Emitting metrics (e.g. length, distribution, etc.).
- Mutating *external* state (counters, queues, caches…) while keeping the wrapped data intact.

**Summary:**

- Reach for `pipe` when your function is part of the **logical pipeline** and returns a new value.
- Reach for `tap` when your function is purely **observational / side-effectful** and should not change the value.

Both live at the *wrapper* level, not at the element level.

---

## 2. Mapping vs Observing Elements: `map`, `inspect`, `for_each`, `peek`

Most of the time you are not transforming the entire wrapper, but **elements inside** it or values wrapped by it (`Some`, `Ok`). There is a clear split between:

- Methods that **transform** values: `map` and friends.
- Methods that **observe** values: `inspect`, `inspect_err`, `for_each`, `peek`.

The key question is:

> “Does my function produce a new value that should replace the old one, or does it just perform a side effect?”

### 2.1 `map` on `Iter`

`Iter.map` takes a function `T -> U` and returns a new `Iter[U]` with transformed elements.

```python
pc.Seq([1, 2, 3]).iter().map(lambda x: x * 2).collect() # -> Seq([2, 4, 6])
pc.Seq(["hello", "world"]).iter().map(str.upper).collect() # -> Seq(["HELLO", "WORLD"])
```

Characteristics:

- **Element-wise transformation**.
- Produces a **new sequence of values** once `collect()` is called.

### 2.2 `map` on `Option` and `Result`

`Option` and `Result` also expose `map`, but now the “element” is the **contained value** (`Some` / `Ok`).

```python
pc.Some("foo").map(len)      # Option[int]
pc.Ok(2).map(lambda x: x * 2)  # Result[int, E]
```

Behaviour:

- If there is a value (`Some`, `Ok`), the function is applied and the result is re-wrapped.
- If there is no value (`NONE`, `Err`), the wrapper is propagated unchanged.

`map` therefore keeps you in the same *error / option* context while allowing you to transform the success value.

### 2.3 Observing without changing: `inspect` and `inspect_err`

Where `map` *changes* the value, `inspect` is the **read-only twin**:

- `Option.inspect(f)` calls `f(value)` if `Some`, then returns the original `Option`.
- `Result.inspect(f)` calls `f(ok_value)` if `Ok`, then returns the original `Result`.
- `Result.inspect_err(f)` calls `f(err_value)` if `Err`, then returns the original `Result`.

This is ideal for logging and debugging paths in a `Result`/`Option` pipeline without breaking the fluent style.

Conceptually:

- `map` = “transform and keep going”.
- `inspect` / `inspect_err` = “peek into the value, but do not touch it”.

### 2.4 Element-wise side effects: `for_each`

On sequences, you sometimes want to run a function on **each element** for its side effects only.

- `Iter.for_each(f)` applies `f` to each element and is **terminal** (returns `None`).
- `{Seq, Dict}.for_each(f)` does the same but returns `self` so that you can continue chaining.

Use `for_each` when:

- You want to push elements into an external sink (e.g. write to a file, send over the network).
- You want to update external state per element (accumulate stats, build a separate index, etc.).

If you need a new sequence of transformed values, use `map` instead.

Iter.for_each is terminal because it **consumes** the iterator. After calling it, there are no more elements to process.

`{Seq, Dict}.for_each` returns self since it calls iter() on an already in-memory structure, leaving the original Seq/Dict intact for further use.

### 2.5 Looking ahead: `peek` on `Iter`

`peek` is specific to `Iter`. It allows you to examine a **prefix** of the iterator without consuming the underlying data:

- It takes `n` and a function `Iterable[T] -> Any`.
- It materialises the first `n` items, passes them to your function, then rebuilds an iterator that yields **all** items (including the peeked ones).

Use cases:

- Log or inspect the “shape” of an iterator early on (first few rows, sample keys, …).
- Implement early-termination logic outside of the chain while keeping lazy behaviour inside.

Compared to `for_each`:

- `for_each` walks the **entire** sequence (and for `Iter` it is terminal).
- `peek` only inspects a **bounded prefix** and returns another `Iter` for continued chaining.

---

## 3. Crossing the Boundary: `into`, `apply`, `from_`

So far we have stayed inside the `pyochain` world. At some point, you either need to:

- **Enter** a chain from raw Python values.
- **Call plain functions** that don’t know about wrappers.
- **Leave** the chain with a final, ordinary Python value.

This is the role of `from_`, `apply`, and `into`.

### 3.1 `from_`: entering the chain with conversion

There are factory-like helpers such as `Iter.from_`, `Seq.from_` and `Option.from_` that convert existing Python values into wrappers. They are convenient, but **not all of them are free**.

Concrete behaviour:

- `Option.from_(value)` is a cheap, explicit conversion from `T | None` to `Option[T]`:
  - `Some(value)` if `value is not None`.
  - `NONE` otherwise.

- `Seq.from_(...)` accepts either an iterable or unpacked values and internally calls `_convert_data` and `_is_sequence`:
  - If you **already** have a `Sequence`, `pc.Seq(existing_sequence)` is cheaper and more direct.
  - `Seq.from_` is mostly a convenience for quick “wrap these values” calls or when you don’t care about the extra checks.

- `Iter.from_(...)` takes an arbitrary iterable and turns it into a lazy `Iter`. It may need to adapt non-iterator iterables into an underlying iterator before wrapping.

In other words, `from_` trades a bit of extra work (type checks / conversions) for ergonomics. In hot paths where you already know you have the right kind of object, prefer the plain constructors:

```python
# Cheaper, when you already know the type
pc.Seq(existing_sequence)      # instead of Seq.from_(existing_sequence)
pc.Iter(existing_iterator)     # instead of Iter.from_(existing_iterator)

# Use from_ when you want convenience or type-normalisation
pc.Seq.from_(1, 2, 3)
pc.Iter.from_(some_iterable)
pc.Option.from_(maybe_value)
```

### 3.2 `apply`: calling plain functions while staying wrapped

`apply` also works on the inner value, but instead of exiting the chain, it **re-wraps** whatever your function returns.

```python
from collections.abc import Sequence
def double_all(xs: Sequence[int]) -> Sequence[int]:
    return [x * 2 for x in xs]

pc.Seq([1, 2, 3]).apply(double_all).map(lambda x: x + 1)
```

Here:

- `double_all` has no dependency on `pyochain`.
- `apply` adapts its input and output for you:
  - It passes the unwrapped `Sequence` to `double_all`.
  - It wraps the result back into a `Seq`.

This is extremely useful to incrementally adopt `pyochain`:

- You can keep most of your logic in ordinary functions that manipulate `list`, `Iterator`, etc.
- You only sprinkle `apply` and `into` at the boundaries, rather than rewriting everything in terms of `Iter`/`Seq`.

Note that `apply` will pass either an `Iterator` or a `Sequence` when used on `Iter` or `Seq` respectively.

Make sure your function can handle the expected type. Often, signatures in codebases are plain `list[T]` even though you do not need any `MutableSequence` features.
Using `apply` is a simple way to promote **immutability** and **genericity** by working with the more general `Sequence` or `Iterator` abstractions.

### 3.3 `into`: leaving the chain

`into` is the symmetric operation: it takes the **inner value** and passes it to a function you provide, returning the raw result.

```python
pc.Seq([1, 2, 3]).into(sum)       # -> 6 (int)
pc.Iter.from_([1, 2, 3]).into(list)  # -> [1, 2, 3] (list[int])
```

You typically use `into` when:

- You want to hand off to a library that expects plain containers.
- You want to compute a final scalar or structure (e.g. `dict`, `DataFrame`, SQL string…).

Once you call `into`, you are **out of the chain**.

**Mental model:**

- `from_`: *wrap this* (enter the chain).
- `apply`: *call my existing function but keep chaining*.
- `into`: *unwrap and give to this function, I’m done*.

---

## 4. Which Types Expose Which Methods?

To summarise by wrapper:

- **All wrappers (`Pipeable`)**
  - `pipe`, `tap`.

- **Sequence-like wrappers (`Iter`, `Seq`, `Dict`)**
  - Boundary methods: `apply`, `inner`, `into`.
  - Transformers: `Iter.map`, `Dict.map_values`, etc.
  - Side-effect / inspection:
    - `Iter`: `for_each` (terminal), `peek`.
    - `Seq`: `for_each` (returns `self`).
    - `Dict`: `for_each` (returns `self`).

- **`Dict`**
  - Follows the same `CommonBase` pattern: `pipe`, `tap`, `apply`, `into`.
  - Exposes iter-style views that in turn support `map`-like operations.

- **`Option` (`Some` / `NONE`)**
  - Transformers: `map`, `and_then`, `map_or`, `map_or_else`, `ok_or`, `ok_or_else`, etc.
  - Observers: `inspect`.

- **`Result` (`Ok` / `Err`)**
  - Transformers: `map`, `map_err`, `and_then`, `or_else`, `transpose`, …
  - Observers: `inspect` (on `Ok`), `inspect_err` (on `Err`).

The common thread is that every family has:

- A **transforming twin** (`map`, `and_then`, …) that changes values.
- An **observing twin** (`inspect`, `for_each`, `peek`) that looks at values but keeps the structure intact.
- Boundary methods (`from_`, `apply`, `into`) to move in and out of the chain.

Once you recognise these patterns, navigating the API—and designing your own helpers in the same style—becomes much more natural.

---

## 5. Impact on Software Design

So far we have focused on *what* each method does and *how* to combine them. The most important consequence of this design is **how it changes the way you structure programs**.

At a high level, `pyochain` encourages you to:

- Keep **concrete domain logic** in small, pure (or mostly pure) functions.
- Express the **orchestration** of your program as a single, readable chain of `pipe` / `apply` / `into` / `from_` (and the usual `map` / `filter` / `for_each`, etc.).

The goal is that your “main” function – or the core of a feature – reads like a pipeline:

```python
def main() -> int:
    return (
        load_raw_data()               # plain function, returns e.g. list[Row]
        .pipe(wrap_as_seq)            # orchestration and normalisation
        .map(normalise_row)           # pure transformations
        .pipe(apply_business_rules)   # may use Option/Result internally
        .apply(group_and_sort)        # call existing helper, stay wrapped
        .into(write_report)           # leave the chain at the very end
    )
```

### 5.1 Orchestration as a single, linear flow

Without chaining, “glue code” tends to look like:

```python
rows = load_raw_data()
rows = [normalise_row(r) for r in rows]
valid_rows = [r for r in rows if is_valid(r)]
grouped = group_and_sort(valid_rows)
return write_report(grouped)
```

This is not terrible, but as complexity grows you quickly accumulate:

- Many intermediate variables (`rows`, `valid_rows`, `grouped`, …).
- Conditionals and error-handling branches interleaved with data transformations.
- State that has to be threaded through multiple steps manually.

With `pyochain`, you can push the *details* into functions and keep the **top-level orchestration** as a single, top-to-bottom chain:

```python
def main() -> int:
    return (
        pc.Seq(load_raw_data())
        .map(normalise_row)
        .filter(is_valid)
        .apply(group_and_sort)
        .into(write_report)
    )
```

You read the flow from top to bottom, with very few names in scope, and every step corresponds to a meaningful stage in the pipeline.

### 5.2 Pure functions first, glue code second

Because `pipe`, `apply` and `into` make it easy to call plain functions, you are encouraged to design **most of your code as normal, testable functions**:

- `normalise_row(row: Row) -> Row`
- `apply_business_rules(row: Row) -> Result[Row, Error]`
- `group_and_sort(rows: Sequence[Row]) -> Sequence[Grouped]`

None of these need to know about `Iter`/`Seq`/`Result` – or only about one of them in a small area. The chain then becomes a thin orchestration layer that wires these pure functions together.

The impact on design:

- It is easy to unit-test the core logic without any wrappers involved.
- The pipeline becomes the place where you decide *when* to turn a `Seq` into an `Iter`, *when* to introduce an `Option` or `Result`, and *when* to exit back to plain values.

### 5.3 Closures as generic building blocks

Because the chaining style works well with small functions, Python **closures** (functions that capture variables from their environment) naturally complement it. They let you build small, specialised callables without committing to any particular implementation detail (no framework, no concrete type required).

There are two common, very generic patterns:

1. **Capture configuration once, reuse everywhere**

   ```python
   from collections.abc import Callable, Sequence


   def make_threshold_filter(threshold: float) -> Callable[[float], bool]:
       def _predicate(value: float) -> bool:
           return value >= threshold

       return _predicate


   def filter_values(values: Sequence[float], threshold: float) -> list[float]:
       keep = make_threshold_filter(threshold)
       return (
           pc.Seq(values)
           .filter(keep)
           .into(list)
       )
   ```

   Here, the chain only sees “a predicate over numbers”. The fact that it depends on a particular `threshold` is encapsulated in the closure returned by `make_threshold_filter`.

2. **Define tiny helpers next to the pipeline**

   ```python
   from collections.abc import Sequence


   def describe_numbers(values: Sequence[int], default_minmax: int) -> str:
     def _format_summary(data: pc.Seq[int]) -> str:
       return (
         f"count={data.count()}, "
         f"min={data.min_or(default_minmax)}, "
         f"max={data.max_or(default_minmax)}"
       )

     return (
       pc.Seq(values)
       .map(abs)
       .tap(lambda s: print("debug:", s.inner()))
       .pipe(_format_summary)
     )
   ```

   Here, `_format_summary` is a closure: it **captures** `default_minmax` from `describe_numbers` and uses it when computing the summary, without needing to pass it explicitly through the chain.

This does **not** mean that classes become obsolete. In many real programs you will mix both approaches:

- Use dataclasses / named tuples / typed dicts to model **data**.
- Use closures and small top-level functions to model **behaviour**, especially when that behaviour is passed as callbacks into `map`, `filter`, `tap`, `pipe`, etc.

The key point is that the chaining style makes it easy to:

- Pass closures wherever a callable is expected (`map`, `filter`, `for_each`, `tap`, `pipe`).
- Keep stateful or configuration-heavy pieces in well-defined structures, without turning every bit of behaviour into a large class.

This keeps the design close to “data + functions” rather than “data + classes with a lot of mutable state”, while still being very expressive.

### 5.4 Named tuples and light data structures shine again

In this style, lightweight, immutable data structures like `namedtuple`, `dataclass(frozen=True)` or simple `TypedDict` become a natural fit:

- They are easy to construct and pass around.
- They work well with `map`/`filter`/`group_by`-style operations.
- They interact nicely with closures: you can capture them, transform them, and keep everything value-oriented.

Rather than introducing classes mostly to carry state between methods, you can:

- Use named tuples / dataclasses to describe **shape of data**.
- Use pipelines and small functions to describe **behaviour over that data**.

### 5.5 Refactoring and debugging as secondary benefits

Once your program is structured as “pure functions + orchestration chain”, other benefits follow almost automatically:

- **Refactoring** becomes local and predictable:
  - Changing the shape of intermediate data usually means adjusting a `map` or `apply`.
  - Moving code into or out of helpers is often just switching between `pipe`, `apply` and `into`.

- **Debugging pipelines** is simpler:
  - Insert `tap`, `inspect`, `inspect_err`, `for_each` or `peek` where needed to observe values without disrupting the flow.
  - Remove these hooks once you’re done; the core design stays the same.

- **Gradual adoption** is natural:
  - You can start by wrapping a single slice of logic in a `Seq`/`Iter` pipeline.
  - Over time, you migrate more of the orchestration into chains while leaving domain logic mostly unchanged.

All of this stems from the same design choice: make the *glue* of your program a linear, chainable data flow, and keep the *meat* of your program in small, focused functions.
