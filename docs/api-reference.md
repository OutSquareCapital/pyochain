# API Reference

This page is the entry point to the **complete** public API documentation.

## Collections

- [`Seq[T]`][pyochain._iter.Seq] — Immutable collections (tuple-backed)
- [`Vec[T]`][pyochain._iter.Vec] — Mutable collections (list-backed)
- [`Set[T]`][pyochain._iter.Set] — Immutable collections (frozenset-backed)
- [`SetMut[T]`][pyochain._iter.SetMut] — Mutable sets (set-backed)
- [`Dict[K, V]`][pyochain._dict.Dict] — Chainable dictionaries

## Iterators

- [`Iter[T]`][pyochain._iter.Iter] — Lazy processing of iterators
- [`Peekable`][pyochain._iter.Peekable] — Peeking iterator
- [`Unzipped`][pyochain._iter.Unzipped] — Unzipped iterator results

## Error handling

- [`Result[T, E]`][pyochain.rs.Result] — Explicit error handling (`Ok` / `Err`)
- [`Ok`][pyochain.rs.Ok]
- [`Err`][pyochain.rs.Err]
- [`ResultUnwrapError`][pyochain.rs.ResultUnwrapError]

## Optional values

- [`Option[T]`][pyochain.rs.Option] — Optional values (`Some` / `NONE`)
- [`Some`][pyochain.rs.Some]
- [`None`][pyochain.rs.NoneOption]
- [`OptionUnwrapError`][pyochain.rs.OptionUnwrapError]

## Traits & mixins

### Fluent Traits

- [`Pipeable`][pyochain.rs.Pipeable] — `.into()`, `.inspect()`
- [`Checkable`][pyochain.rs.Checkable] — `.then()`, `.ok_or()`, ...

### Abstract Collection Traits

- [`PyoIterable[T]`][pyochain.traits._iterable.PyoIterable] — Base trait for all iterables
- [`PyoIterator[T]`][pyochain.traits._iterable.PyoIterator] — Iterator trait
- [`PyoCollection[T]`][pyochain.traits._iterable.PyoCollection] — Base trait for eager collections
- [`PyoSequence[T]`][pyochain.traits._iterable.PyoSequence] — Sequence trait
- [`PyoMutableSequence[T]`][pyochain.traits._iterable.PyoMutableSequence] — Mutable sequence trait
- [`PyoSet[T]`][pyochain.traits._iterable.PyoSet] — Set trait
- [`PyoMappingView[T]`][pyochain.traits._iterable.PyoMappingView] — Mapping view trait
- [`PyoMapping[K, V]`][pyochain.traits._iterable.PyoMapping] — Mapping trait
- [`PyoMutableMapping[K, V]`][pyochain.traits._iterable.PyoMutableMapping] — Mutable mapping trait

### Mapping Views

- [`PyoKeysView[K]`][pyochain.traits._iterable.PyoKeysView] — Keys view
- [`PyoValuesView[V]`][pyochain.traits._iterable.PyoValuesView] — Values view
- [`PyoItemsView[K, V]`][pyochain.traits._iterable.PyoItemsView] — Items view
