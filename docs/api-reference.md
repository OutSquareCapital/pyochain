# API Reference

This page is the entry point to the **complete** public API documentation.

## Collections

- [`Seq[T]`](reference/seq.md) — Immutable Sequence (`tuple`-backed)
- [`Vec[T]`](reference/vec.md) — Mutable Sequence (`list`-backed)
- [`Set[T]`](reference/set.md) — Immutable Set (`frozenset`-backed)
- [`SetMut[T]`](reference/setmut.md) — Mutable Set (`set`-backed)
- [`Dict[K, V]`](reference/dict.md) — Mutable Mapping (`dict`-backed)
- [`Range`](reference/range.md) —  Integer ranges (`range`-backed)

## Iterators

- [`Iter[T]`](reference/iter.md) — Lazy processing of iterators

## Error handling

- [`Result[T, E]`](reference/result.md) — Actual type to use for explicit error handling (`Ok` / `Err`). Used for type hints.
- [`ResultType[T, E]`](reference/resulttype.md) — Common interface for `Ok` and `Err`. Reference their methods.
- [`Ok[T]`](reference/ok.md) - `Ok` variant of `Result`
- [`Err[E]`](reference/err.md) - `Err` variant of `Result`
- [`ResultUnwrapError`](reference/resultunwraperror.md) - Exception raised when unwrapping a `Result` that is an `Err`

## Optional values

- [`Option[T]`](reference/option.md) — Optional values (`Some` / `NONE`)
- [`Some[T]`](reference/some.md) - `Some` variant of `Option`
- [`NONE`](reference/null.md) - `NONE` variant of `Option`. Constant singleton.
- [`OptionUnwrapError`](reference/optionunwraperror.md) - Exception raised when unwrapping an `Option` that is `NONE`

## Mixins & ABC's

### Fluent Mixins

- [`Pipeable`](reference/pipeable.md) — Fluent methods for chaining operations
- [`Checkable`](reference/checkable.md) — Conversion to `Option`/`Result` based on instance truthiness

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
