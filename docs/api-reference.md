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

### Fluent Traits

- [`Pipeable`](reference/pipeable.md) — `.into()`, `.inspect()`
- [`Checkable`](reference/checkable.md) — `.then()`, `.ok_or()`, ...

### Abstract Collection Traits

- [`PyoIterable[T]`](reference/pyoiterable.md) — Base trait for all iterables
- [`PyoIterator[T]`](reference/pyoiterator.md) — Iterator trait
- [`PyoCollection[T]`](reference/pyocollection.md) — Base trait for eager collections
- [`PyoSequence[T]`](reference/pyosequence.md) — Sequence trait
- [`PyoMutableSequence[T]`](reference/pyomutablesequence.md) — Mutable sequence trait
- [`PyoSet[T]`](reference/pyoset.md) — Set trait
- [`PyoMutableSet[T]`](reference/pyomutableset.md) — Mutable set trait
- [`PyoMappingView[T]`](reference/pyomappingview.md) — Mapping view trait
- [`PyoMapping[K, V]`](reference/pyomapping.md) — Mapping trait
- [`PyoMutableMapping[K, V]`](reference/pyomutablemapping.md) — Mutable mapping trait

### Mapping Views

- [`PyoKeysView[K]`](reference/pyokeysview.md) — Keys view
- [`PyoValuesView[V]`](reference/pyovaluesview.md) — Values view
- [`PyoItemsView[K, V]`](reference/pyoitemsview.md) — Items view
