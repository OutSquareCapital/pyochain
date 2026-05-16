# API Reference

This page is the entry point to the **complete** public API documentation.

## Collections

- [`Seq[T]`](reference/seq.md) — Immutable collections (tuple-backed)
- [`Vec[T]`](reference/vec.md) — Mutable collections (list-backed)
- [`Set[T]`](reference/set.md) — Immutable collections (frozenset-backed)
- [`SetMut[T]`](reference/setmut.md) — Mutable sets (set-backed)
- [`Dict[K, V]`](reference/dict.md) — Chainable dictionaries

## Iterators

- [`Iter[T]`](reference/iter.md) — Lazy processing of iterators
- [`Peekable[T]`](reference/peekable.md) — Peeking iterator
- [`Unzipped[T]`](reference/unzipped.md) — Unzipped iterator results
- [`Range`](reference/range.md) — Lazy integer ranges

## Error handling

- [`Result[T, E]`](reference/result.md) — Explicit error handling (`Ok` / `Err`)
- [`Ok[T]`](reference/ok.md)
- [`Err[E]`](reference/err.md)
- [`ResultUnwrapError`](reference/resultunwraperror.md)

## Optional values

- [`Option[T]`](reference/option.md) — Optional values (`Some` / `NONE`)
- [`Some[T]`](reference/some.md)
- [`NONE`](reference/null.md)
- [`OptionUnwrapError`](reference/optionunwraperror.md)

## Mixins & ABC's

### Fluent Mixins

- [`Pipeable`](reference/pipeable.md) — `.into()`, `.inspect()`
- [`Checkable`](reference/checkable.md) — `.then()`, `.ok_or()`, ...

### Abstract Base Classes

- [`PyoIterable[T]`](reference/pyoiterable.md) — Base ABC for all iterables
- [`PyoIterator[T]`](reference/pyoiterator.md) — Iterator ABC
- [`PyoCollection[T]`](reference/pyocollection.md) — Base ABC for eager collections
- [`PyoSequence[T]`](reference/pyosequence.md) — Sequence ABC
- [`PyoMutableSequence[T]`](reference/pyomutablesequence.md) — Mutable sequence ABC
- [`PyoSet[T]`](reference/pyoset.md) — Set ABC
- [`PyoMappingView[T]`](reference/pyomappingview.md) — Mapping view ABC
- [`PyoMapping[K, V]`](reference/pyomapping.md) — Mapping ABC
- [`PyoMutableMapping[K, V]`](reference/pyomutablemapping.md) — Mutable mapping ABC

### Mapping Views

- [`PyoKeysView[K]`](reference/pyokeysview.md) — Keys view
- [`PyoValuesView[V]`](reference/pyovaluesview.md) — Values view
- [`PyoItemsView[K, V]`](reference/pyoitemsview.md) — Items view
