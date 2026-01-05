# API Reference

This page is the entry point to the **complete** public API documentation.

## Collections

- [`Iter[T]`](reference/iter.md) — Lazy processing of iterators
- [`Seq[T]`](reference/seq.md) — Immutable collections (tuple-backed)
- [`Vec[T]`](reference/vec.md) — Mutable collections (list-backed)
- [`Set[T]`](reference/set.md) — Immutable collections (frozenset-backed)
- [`SetMut[T]`](reference/setmut.md) — Mutable sets (set-backed)
- [`Dict[K, V]`](reference/dict.md) — Chainable dictionaries

## Error handling

- [`Result[T, E]`](reference/result.md) — Explicit error handling (`Ok` / `Err`)
- [`Ok`](reference/ok.md)
- [`Err`](reference/err.md)

## Optional values

- [`Option[T]`](reference/option.md) — Optional values (`Some` / `NONE`)
- [`Some`](reference/some.md)
- [`None`](reference/none.md)

## Traits & mixins

- [`Pipeable`](reference/pipeable.md) — `.into()`, `.inspect()`
- [`Checkable`](reference/checkable.md) — `.then()`, `.ok_or()`, ...
