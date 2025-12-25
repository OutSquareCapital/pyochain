# Chaining: Passing Values Between Functions

`pyochain` is built around the idea of **staying in a fluent chain** while still being able to call your own functions, observe what happens, and eventually get back to plain Python values.

This guide explains how data flows between:

- the wrappers (`Iter`, `Seq`, `Vec`, `Set`, `Dict`, `Result`, `Option`, …),
- your own functions (pure or not),
- and the outside world (final values, side effects).

We will mostly classify methods by **behaviour**, not by concrete type.

## Behavioural categories

- Boundary conversion (in/out): constructors, `from_`, `into`, `collect`, `inner`, `unwrap`.
- Side-effect hooks (observe without changing): `inspect`, `inspect_err`, `for_each`, `peek`.
- Pure transformations (produce new values): `map`, `and_then`, `map_err`, `ok_or`, `or_else`.

Along the way, we also highlight **which wrappers expose what**, so you can quickly see what is available on `Iter` vs `Seq` vs `Vec` vs `Dict` vs `Result` vs `Option`, and how they fit together.

---

## 1. Boundary Conversion: constructors, `from_`, `into`, `collect`, `inner`

At the boundary of a chain you often want to:

- enter from plain Python values,
- call your own helpers (sometimes wrapper-aware, sometimes not),
- materialise / unwrap, and exit.

In the current API, `from_` and `into` are the two symmetric “conversion” primitives:

- `from_` converts *Python values → wrapper*.
- `into` converts *wrapper → whatever your function returns* (another wrapper or a plain value).

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

Since `Iter` implements the `Iterator` protocol and `Seq` implements the `Sequence` protocol, wrappers can be passed directly to ordinary Python functions:

```python
def compute_data(data: Iterable[int]) -> int:
    return sum(x * 2 for x in data)

pc.Iter((1, 2, 3)).into(compute_data)  # Iter[int] -> int
```

Use cases:

- Implement branching logic (e.g. return `Option` from `Seq`, or turn a `Result` into another `Result`).
- If your helper returns a `pyochain` wrapper, you stay in the chain.
- If your helper returns a plain value, you have exited the chain.

### 1.2 `from_`: convert in

`from_` methods are convenience constructors. They accept unpacked values or iterables, but involve extra checks and potential materialisation.

**General pattern:**

- If the native type matches (e.g. `list` for `Vec`, `tuple` for `Seq`, `set`/`frozenset` for `Set`), wraps directly.
- Otherwise, materialises the iterable into the target type.
- Unpacked values are always materialised.

```python
import pyochain as pc

# Iter.from_: mainly for unpacked values or string normalisation
pc.Iter.from_(1, 2, 3)         # -> Iter over (1, 2, 3)
pc.Iter.from_("hello")         # -> Iter over the words ("hello",)
pc.Iter("hello")              # -> Iter over the characters ('h', 'e', 'l', 'l', 'o')

# Seq.from_: wraps Sequence directly, else materialises into tuple
pc.Seq.from_([1, 2, 3])        # -> Seq((1, 2, 3))
pc.Seq.from_(1, 2, 3)          # -> Seq((1, 2, 3))

# Vec.from_: wraps list directly, else materialises into list
pc.Vec.from_([1, 2, 3])        # -> Vec([1, 2, 3]) (no copy)
pc.Vec.from_((1, 2, 3))        # -> Vec([1, 2, 3]) (materialises)

# Set.from_: wraps set/frozenset directly, else materialises into frozenset
pc.Set.from_({1, 2, 3})        # -> Set({1, 2, 3})
pc.Set.from_(1, 2, 3)          # -> Set(frozenset({1, 2, 3}))
pc.Option.from_(42)            # -> Some(42)
pc.Option.from_(None)          # -> NONE
```

**When to use `from_` vs direct constructor:**

- Use the direct constructor (`Iter(iterable)`, `Seq(sequence)`, ...) when you already have the right type.
- Use `from_` for unpacked values or when you want automatic conversion.
- For `Iter`, prefer `collect()` over `.into(Seq.from_)` as it's more direct.

### 1.3 `collect`: materialise an `Iter`

`collect` is how you exit from lazy iteration back to an in-memory collection.

**Signature:** `collect(collector: Collector[T] = tuple) -> Seq[T] | Vec[T] | Set[T]`

**Supported collectors:**

- `tuple` (default) → `Seq[T]` (immutable sequence)
- `list` → `Vec[T]` (mutable list)
- `set` → `Set[T]` (unordered unique elements)
- `frozenset` → `Set[T]` (immutable unordered unique elements)

You can pass the type directly (`list`, `tuple`, `set`, `frozenset`) or any callable that takes an `Iterable[T]` and returns one of those types.

```python
import pyochain as pc

# Default: tuple -> Seq
pc.Iter(range(5)).collect()              # -> Seq(0, 1, 2, 3, 4)

# Explicit: list -> Vec
pc.Iter(range(5)).collect(list)          # -> Vec([0, 1, 2, 3, 4])

# Explicit: set -> Set (removes duplicates)
pc.Iter([1, 2, 2, 3]).collect(set)       # -> Set({1, 2, 3})

# Explicit: frozenset -> Set
pc.Iter([1, 2, 2, 3]).collect(frozenset) # -> Set(frozenset({1, 2, 3}))
```

**For other types** (polars, pandas, numpy), use `.into(DataFrame)` or similar.

#### `Option.from_`

`Option.from_` wraps the common "maybe" convention (`T | None`):

```python
import pyochain as pc

def find_user_id(name: str) -> int | None:
    return 123 if name == "alice" else None

pc.Option.from_(find_user_id("alice")).map(lambda uid: uid + 1)
```

---

## 2. Transforming Values: `map` vs `for_each` (especially on `Iter`)

Most of the time you are not transforming the entire wrapper, but **elements inside** it.

The key question (especially for `Iter`) is:

> “Am I producing new elements (`map`), or just running a side effect (`for_each`)?”

### 2.1 `map` on `Iter`

`Iter.map` takes a function `T -> U` and returns a new `Iter[U]` with transformed elements.

```python
pc.Iter((1, 2, 3)).map(lambda x: x * 2).collect() # -> Seq((2, 4, 6))
pc.Iter(("hello", "world")).map(str.upper).collect() # -> Seq(("HELLO", "WORLD"))
```

Characteristics:

- **Element-wise transformation**.
- Produces a **new sequence of values** once `collect()` is called.

`Option` and `Result` also expose `map` (and other transforms), but that is covered in Section 4.

### 2.2 `for_each` on `Iter`

`Iter.for_each(f)` applies `f` to each element for side effects only and is **terminal** (returns `None`). It consumes the iterator completely – equivalent to a classic Python for-loop.

---

## 3. Side-Effect Hooks: `inspect`, `inspect_err`, `for_each`, `peek`

These methods are all about **observing** values and triggering side effects (logging, metrics, IO) while keeping the dataflow chained.

- `inspect`: observe self (available on all wrappers).
- `inspect_err`: observe the error value inside `Result`.
- `for_each`: run effects per element for `Iter` only (terminal).
- `peek`: inspect a bounded prefix of an `Iter` without consuming it.

If your callback returns something meaningful and should replace the old value, you probably want a transform (`map`, `and_then`, ...), not a side-effect hook.

### 3.1 `inspect`: observe without changing

`inspect` receives the wrapper, runs a side-effect, and returns the **same instance** unchanged. Where `map` transforms, `inspect` is the **read-only twin**.

Common use cases: logging, writing to a file, or sending the value over the network.

```python
import pyochain as pc

pc.Seq((1, 2, 3)).inspect(print).last()  # logs the whole Seq
```

### 3.2 `inspect` and `inspect_err`: observe `Option`/`Result` values

`inspect` on `Option`/`Result` is **value-aware**:

- `Option.inspect(f)` calls `f(value)` only if `Some`, then returns the original `Option`.
- `Result.inspect(f)` calls `f(ok_value)` only if `Ok`, then returns the original `Result`.
- `Result.inspect_err(f)` calls `f(err_value)` only if `Err`, then returns the original `Result`.

This is ideal for logging and debugging paths in a `Result`/`Option` pipeline without breaking the fluent style.

```python
import pyochain as pc

def _log_val(n: int) -> None:
    print(f"Parsed int: {n}")

def process_value(s: str) -> pc.Result[int, ValueError]:
    return (
        pc.Option.from_(int(s) if s.isdigit() else None)
        .ok_or(ValueError("not an int"))
        .inspect(_log_val)
    )

process_value("42")  # logs "Parsed int: 42"
process_value("foo")  # does not log anything
```

### 3.3 `peek` on `Iter`: inspect a prefix without consuming

`peek` is specific to `Iter`. It allows you to examine a **prefix** of the iterator without consuming the underlying data:

- It takes `n` and a function `Iterable[T] -> Any`.
- It materialises the first `n` items, passes them to your function, then rebuilds an iterator that yields **all** items (including the peeked ones).

Compared to `for_each`:

- `for_each` walks the **entire** sequence and is terminal.
- `peek` only inspects a **bounded prefix** and returns another `Iter` for continued chaining.

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
        .inspect(lambda s: print("debug:", s.inner()))
        .into(_format_summary)
    )
```

Here, `_format_summary` is a closure: it **captures** `default_minmax` from `describe_numbers` and uses it when computing the summary, without needing to pass it explicitly through the chain.

This does **not** mean that classes become obsolete. In many real programs you will mix both approaches:

- Use dataclasses / named tuples / typed dicts to model **data**.
- Use closures and small top-level functions to model **behaviour**.
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
  - Insert `inspect`, `inspect_err`, `for_each` or `peek` where needed to observe values without disrupting the flow.
  - Remove these hooks once you’re done; the core design stays the same.

- **Gradual adoption** is natural:
  - You can start by wrapping a single slice of logic in a `Seq`/`Iter` pipeline.
  - Over time, you migrate more of the orchestration into chains while leaving domain logic mostly unchanged.

All of this stems from the same design choice: make the *glue* of your program a linear, chainable data flow, and keep the *meat* of your program in small, focused functions.

---

**Key take-aways:**

- Every wrapper has `into` and `inspect` (inherited from `Pipeable`).
- `Iter` is an `Iterator`, `Seq` is a `Sequence`, `Vec` is a `MutableSequence`, and `Set` is a `Collection`, so many plain-Python functions can consume them directly (and `into` is the ergonomic way to express that in a chain).
- `Option` and `Result` share transform methods (`map`, `and_then`, `or_else`).
- `Iter.for_each` is **terminal** (consumes the iterator and returns `None`).
