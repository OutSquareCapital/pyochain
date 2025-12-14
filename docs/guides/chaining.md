# Chaining: Passing Values Between Functions

`pyochain` is built around the idea of **staying in a fluent chain** while still being able to call your own functions, observe what happens, and eventually get back to plain Python values.

This guide explains how data flows between:

- the wrappers (`Iter`, `Seq`, `Dict`, `Result`, `Option`, …),
- your own functions (pure or not),
- and the outside world (final values, side effects).

We will mostly classify methods by **behaviour**, not by concrete type.

## Behavioural categories

- Boundary conversion (in/out): constructors, `from_`, `into`, `collect`, `inner`, `unwrap`.
- Side-effect hooks (observe without changing): `tap`, `inspect`, `inspect_err`, `for_each`, `peek`.
- Pure transformations (produce new values): `map`, `and_then`, `map_err`, `ok_or`, `or_else`.

Along the way, we also highlight **which wrappers expose what**, so you can quickly see what is available on `Iter` vs `Seq` vs `Result` vs `Option`, and how they fit together.

---

## 1. Boundary Conversion: constructors, `from_`, `into`, `collect`, `inner`

At the boundary of a chain you often want to:

- enter from plain Python values,
- call your own helpers (sometimes wrapper-aware, sometimes not),
- materialise / unwrap, and exit.

In the current API, `from_` and `into` are the two symmetric “conversion” primitives:

- `from_` converts *Python values → wrapper*.
- `into` converts *wrapper → whatever your function returns* (another wrapper or a plain value).

Because `Iter` implements the `Iterator` protocol and `Seq` implements the `Sequence` protocol, wrappers can often be consumed directly by ordinary Python functions.

### 1.1 `into`: convert out / delegate (and optionally keep chaining)

`into` passes the current wrapper instance as the first argument to your function and returns **whatever the function returns**.

Conceptually, `into` lets you *name* (or inline) a part of your chain:

```python
import pyochain as pc


def maybe_sum(xs: pc.Seq[int]) -> pc.Option[int]:
    if xs.length() == 0:
        return pc.NONE
    return pc.Some(xs.sum())

pc.Seq(range(5)).into(maybe_sum)  # Seq[int] -> Option[int]
```

But since `Iter` implements the `Iterator` protocol and `Seq` implements the `Sequence` protocol, you can also pass them to ordinary Python functions that expect those protocols:

```python
def compute_data(data: Iterable[int]) -> int:
    """Example function that works on an Iterable[int].
    Most of the time, this will be (if typed at all!) as list[int] or tuple[int, ...].
    This is also often a good opportunity to implement more generic Protocols (here, only __iter__ is needed really, so Iterable[int] is sufficient).
    """
    return sum(x * 2 for x in data)

pc.Iter((1, 2, 3)).into(compute_data)  # Iter[int] -> int
```

You can think of `into` as "Give this value to this function who will convert it into this other value".

- Use it to implement branching logic (e.g. return `Option` from `Seq`, or turn a `Result` into another `Result`).
- If your helper returns a `pyochain` wrapper, you stay in the `pyochain` world.
- If your helper returns a plain value, you have exited the chain.

### 1.2 `from_`: convert in (and what is cheap vs not)

`from_` methods are convenience constructors. Some are intentionally “not cheap” because they normalise inputs.

#### `Iter.from_`

Prefer `pc.Iter(iterable)` for an existing iterable.
Use `Iter.from_(...)` mainly for the ergonomics of **unpacked values**, or when dealing with strings for example.

```python
import pyochain as pc
pc.Iter((1,), (2,), (3,))      # .from_ is more readable here
pc.Iter.from_(1, 2, 3)         # unpacked
pc.Iter([1, 2, 3])             # iterable (preferred if already computed)
pc.Iter.from_("hello")        # from_ normalises strings into char iterators, avoid splitting it manually.
```

#### `Seq.from_`

`Seq.from_` do the same, but since it's a more precise wrapper, it has to do a few checks:

- If you pass a `Sequence`, it can wrap it directly.
- If you pass a non-`Sequence` iterable (set, iterator, generator, ...), it will **collect/materialise** it (currently into a `tuple`).
- If you pass unpacked values, it will also materialise them into a `tuple`.

That means `Seq.from_` is not “cheap” in the general case because it has to do a few checks.

Basically, those two are equivalent:

```python
import pyochain as pc

pc.Iter(range(3)).collect()            # -> Seq((0, 1, 2))
pc.Iter(range(3)).into(pc.Seq.from_)   # -> Seq((0, 1, 2))
```

They both materialise the iterator into a `Seq`. `collect()` avoids the extra “is this iterable?” check that `Seq.from_` does, so `collect()` is the more direct materialisation primitive.

#### `Option.from_`

`Option.from_` is extremely useful in most situations because it wraps the very common “maybe” convention (`T | None`) into a chain-friendly shape:

```python
import pyochain as pc


def find_user_id(name: str) -> int | None:  # legacy / external API
    return 123 if name == "alice" else None


pc.Option.from_(find_user_id("alice")).map(lambda user_id: user_id + 1)
```

This is often the simplest way to turn existing Python functions into composable pipelines.

---

## 2. Side-Effect Hooks: `tap`, `inspect`, `inspect_err`, `for_each`, `peek`

These methods are all about **observing** values and triggering side effects (logging, metrics, IO) while keeping the dataflow chained.

- `tap`: observe self (available on all wrappers).
- `inspect`: observe the success value inside `Option`/`Result`.
- `inspect_err`: observe the error value inside `Result`.
- `for_each`: run effects per element for `Iter`/`Seq`/`Dict`.
- `peek`: inspect a bounded prefix of an `Iter` without consuming it.

If your callback returns something meaningful and should replace the old value, you probably want a transform (`map`, `and_then`, ...), not a side-effect hook.

### 2.1 `tap`: side effects on the wrapper

`tap` receives the wrapper, runs a side-effect, and returns the **same instance** unchanged.
This can be very convenient for example when dealing with functions that need the value, return None by design, but you still need the same value later.
printing/logging being an obvious use case, but writing to a file or sending the value over the network are also common examples.

```python
import pyochain as pc

pc.Seq((1, 2, 3)).tap(print).last()  # logs the whole Seq
```

### 2.2 `inspect` and `inspect_err`: observe `Option`/`Result` values

Where `map` changes the value, `inspect` is the **read-only twin**:

- `Option.inspect(f)` calls `f(value)` if `Some`, then returns the original `Option`.
- `Result.inspect(f)` calls `f(ok_value)` if `Ok`, then returns the original `Result`.
- `Result.inspect_err(f)` calls `f(err_value)` if `Err`, then returns the original `Result`.

This is ideal for logging and debugging paths in a `Result`/`Option` pipeline without breaking the fluent style.

### 2.3 `peek` on `Iter`: inspect a prefix without consuming

`peek` is specific to `Iter`. It allows you to examine a **prefix** of the iterator without consuming the underlying data:

- It takes `n` and a function `Iterable[T] -> Any`.
- It materialises the first `n` items, passes them to your function, then rebuilds an iterator that yields **all** items (including the peeked ones).

Compared to `for_each`:

- `for_each` walks the **entire** sequence (and for `Iter` it is terminal).
- `peek` only inspects a **bounded prefix** and returns another `Iter` for continued chaining.

---

## 3. Transforming Values: `map` vs `for_each` (especially on `Iter`)

Most of the time you are not transforming the entire wrapper, but **elements inside** it.

The key question (especially for `Iter`) is:

> “Am I producing new elements (`map`), or just running a side effect (`for_each`)?”

### 3.1 `map` on `Iter`

`Iter.map` takes a function `T -> U` and returns a new `Iter[U]` with transformed elements.

```python
pc.Iter((1, 2, 3)).map(lambda x: x * 2).collect() # -> Seq((2, 4, 6))
pc.Iter(("hello", "world")).map(str.upper).collect() # -> Seq(("HELLO", "WORLD"))
```

Characteristics:

- **Element-wise transformation**.
- Produces a **new sequence of values** once `collect()` is called.

`Option` and `Result` also expose `map` (and other transforms), but that is covered in Section 4.

### 3.2 `for_each` on `Iter`

On Sequences, you sometimes want to run a function on **each element** for its side effects only.

- `Iter.for_each(f)` applies `f` to each element and is **terminal** (returns `None`).
- `{Seq, Dict}.for_each(f)` does the same but returns `self` so that you can continue chaining.

Iter.for_each is terminal because it **consumes** the iterator. After calling it, there are no more elements to process.

`{Seq, Dict}.for_each` returns self since it calls iter() on an already in-memory structure, leaving the original Seq/Dict intact for further use.
This is best compared to a classic python for-loop over a list or dict who would print the values.
`Seq.tap(lambda x: x.iter().map(lambda v: print(v)))` would be an equivalent, altough much less ergonomic.

---

## 4. Chaining `Option` and `Result`: `map`, `and_then`, `ok_or`

`Option` and `Result` are the “control-flow” wrappers: they let you express absence and failure without breaking the chain.

### 4.1 `map` vs `and_then`

- Use `map` when your function returns a plain value ($T \to U$).
- Use `and_then` when your function already returns an `Option`/`Result` and you know you want to work with the `Some`/`Ok` case only.

```python
import pyochain as pc


def parse_int(s: str) -> pc.Option[int]:
    return pc.Option.from_(int(s)) if s.isdigit() else pc.NONE


pc.Some("12").and_then(parse_int).map(lambda n: n + 1) # -> Some(value=13)
```

### 4.2 `ok_or`: turn an `Option` into a `Result`

When “missing” should become an explicit error, convert:

```python
import pyochain as pc

pc.Some(1).ok_or(ValueError("missing"))  # -> Ok(1)
pc.NONE.ok_or(ValueError("missing"))  # -> Err(ValueError("missing"))
```

From there you can keep chaining on `Result` (`map`, `and_then`, `map_err`, `inspect_err`, ...).

---

## 5. Impact on Software Design

So far we have focused on *what* each method does and *how* to combine them. The most important consequence of this design is **how it changes the way you structure programs**.

At a high level, `pyochain` encourages you to:

- Keep **concrete domain logic** in small, pure (or mostly pure) functions.
- Express the **orchestration** of your program as a single, readable chain of `into` / `tap` (and the usual `map` / `filter` / `for_each`, etc.).

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

With `pyochain`, you can push the *details* into functions and keep the **top-level orchestration** as a single, top-to-bottom chain.
You read the flow from top to bottom, with very few names in scope, and every step corresponds to a meaningful stage in the pipeline.

The goal is that your “main” function – or the core of a feature – reads like a pipeline:

```python
def main() -> int:
    return (
        pc.Iter(load_raw_data())       # wrap once
        .map(normalise_row)           # pure transformations
        .filter(is_valid)
        .collect()                    # materialise when needed
        .into(group_and_sort)         # delegate to a helper
        .into(write_report)           # leave the chain at the very end
    )
```

### 5.2 Pure functions first, glue code second

Because `into` makes it easy to call wrapper-aware helpers, you are encouraged to design **most of your code as normal, testable functions**:

- `normalise_row(row: Row) -> Row`
- `apply_business_rules(row: Row) -> Result[Row, Error]`
- `group_and_sort(rows: Iterable[Row]) -> Seq[Grouped]`

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

import pyochain as pc

def _make_threshold_filter(threshold: float) -> Callable[[float], bool]:
    def _predicate(value: float) -> bool:
        return value >= threshold

    return _predicate

def filter_values(values: Sequence[float], threshold: float) -> pc.Seq[float]:
    return pc.Seq(values).iter().filter(_make_threshold_filter(threshold)).collect()

```

Here, the chain only sees “a predicate over numbers”. The fact that it depends on a particular `threshold` is encapsulated in the closure returned by `make_threshold_filter`.

1. **Define tiny helpers next to the pipeline**

```python
import pyochain as pc
from collections.abc import Sequence


def describe_numbers(values: Sequence[int], default_minmax: int) -> str:
    def _format_summary(data: pc.Seq[int]) -> str:
        if data.length() == 0:
            return f"length=0, default={default_minmax}"
        return f"length={data.length()}, min={data.min()}, max={data.max()}"

    return (
        pc.Iter(values)
        .map(abs)
        .collect()
        .tap(lambda s: print("debug:", s.inner()))
        .into(_format_summary)
    )
```

Here, `_format_summary` is a closure: it **captures** `default_minmax` from `describe_numbers` and uses it when computing the summary, without needing to pass it explicitly through the chain.

This does **not** mean that classes become obsolete. In many real programs you will mix both approaches:

- Use dataclasses / named tuples / typed dicts to model **data**.
- Use closures and small top-level functions to model **behaviour**, especially when that behaviour is passed as callbacks into `map`, `filter`, `for_each`, `tap`, `into`, etc.

The key point is that the chaining style makes it easy to:

- Pass closures wherever a callable is expected (`map`, `filter`, `for_each`, `tap`, `into`).
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
  - Changing the shape of intermediate data usually means adjusting a `map` or an `into(...)` helper.
  - Moving code into or out of helpers is often just turning an inline step into an `into(helper)` call.

- **Debugging pipelines** is simpler:
  - Insert `tap`, `inspect`, `inspect_err`, `for_each` or `peek` where needed to observe values without disrupting the flow.
  - Remove these hooks once you’re done; the core design stays the same.

- **Gradual adoption** is natural:
  - You can start by wrapping a single slice of logic in a `Seq`/`Iter` pipeline.
  - Over time, you migrate more of the orchestration into chains while leaving domain logic mostly unchanged.

All of this stems from the same design choice: make the *glue* of your program a linear, chainable data flow, and keep the *meat* of your program in small, focused functions.

---

**Key take-aways:**

- Every wrapper has `into` and `tap` (inherited from `Pipeable`).
- `Iter` is an `Iterator` and `Seq` is a `Sequence`, so many plain-Python functions can consume them directly (and `into` is the ergonomic way to express that in a chain).
- `Option` and `Result` share transform methods (`map`, `and_then`, `or_else`) and the observe method `inspect`.
- `Iter.for_each` is **terminal** (consumes the iterator); `Seq.for_each` and `Dict.for_each` return `self`.
