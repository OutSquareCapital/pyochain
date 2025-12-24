# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

Welcome to the `pyochain` documentation! This library brings a fluent, declarative API inspired by Rust and DataFrame libraries to your Python iterables and dictionaries.

## Quick Start

```bash
uv add pyochain
```

```python
import pyochain as pc

# Lazy processing with Iter
pc.Iter.from_count(1).filter(lambda x: x % 2 != 0).map(lambda x: x ** 2).take(5).collect()
# → Seq(1, 9, 25, 49, 81)

# Type-safe error handling with Result
def divide(a: int, b: int) -> pc.Result[float, str]:
    return pc.Err("Division by zero") if b == 0 else pc.Ok(a / b)

divide(10, 0).unwrap_or(0.0)  # → 0.0
```

## Documentation Navigation

### Guides

- [**Chaining Guide**](guides/chaining.md) — Master the art of chaining with pyochain

### API Reference

#### Collections

- [**Iter[T]**](reference/iter.md) — Lazy processing of iterators
- [**Seq[T]**](reference/seq.md) — Immutable collections (tuple-backed)
- [**Set[T]**](reference/set.md) — Immutable collections (set|frozenset-backed)
- [**Vec[T]**](reference/vec.md) — Mutable collections (list-backed)
- [**Dict[K, V]**](reference/dict.md) — Chainable dictionaries

#### Error Handling & Optionals

- [**Result[T, E]**](reference/result.md) — Explicit error handling (`Ok` / `Err`)
- [**Option[T]**](reference/option.md) — Optional values (`Some` / `NONE`)

Each reference page includes detailed examples and complete type signatures.

## Philosophy in Brief

- **Declarative** → Replace loops with high-level operations
- **Type-safe** → Generics and overloads for optimal developer experience
- **Lazy & Eager** → `Iter` for efficiency, `Seq`/`Vec` for materialization
- **Fluent chaining** → Compose simple, reusable transformations

For more details on philosophy, inspirations, and dependencies, see the [README](https://github.com/OutSquareCapital/pyochain).

## Useful Links

- [**GitHub Repository**](https://github.com/OutSquareCapital/pyochain)
- [**Contributing Guide**](https://github.com/OutSquareCapital/pyochain/blob/master/CONTRIBUTING.md)
- [**Examples**](https://github.com/OutSquareCapital/pyochain/blob/master/EXAMPLES.md)
- [**PyPI Package**](https://pypi.org/project/pyochain/)
