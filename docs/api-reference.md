# API Reference

This page is the entry point to the **complete** public API documentation.

## Mixins

- [`Tap`](reference/tap.md)
- [`Pipe`](reference/pipe.md)
- [`Fluent`](reference/fluent.md)
- [`Checkable`](reference/checkable.md)

## ABCs

- [`PyoIterable[T]`](reference/pyoiterable.md)
- [`PyoIterator[T]`](reference/pyoiterator.md)
- [`PyoCollection[T]`](reference/pyocollection.md)
- [`PyoSequence[T]`](reference/pyosequence.md)
- [`PyoMutableSequence[T]`](reference/pyomutablesequence.md)
- [`PyoSet[T]`](reference/pyoset.md)
- [`PyoMappingView[T]`](reference/pyomappingview.md)
- [`PyoMapping[K, V]`](reference/pyomapping.md)
- [`PyoMutableMapping[K, V]`](reference/pyomutablemapping.md)

### Concrete Collections & Iterators

### Collections

- [`Seq[T]`](reference/seq.md)
- [`Vec[T]`](reference/vec.md)
- [`Set[T]`](reference/set.md)
- [`SetMut[T]`](reference/setmut.md)
- [`Dict[K, V]`](reference/dict.md)
- [`Range`](reference/range.md)
- [`SliceView[T]`](reference/sliceview.md)
- [`StableSet[T]`](reference/stableset.md)

### Mapping Views

- [`PyoKeysView[K]`](reference/pyokeysview.md)
- [`PyoValuesView[V]`](reference/pyovaluesview.md)
- [`PyoItemsView[K, V]`](reference/pyoitemsview.md)

### Iterators

- [`Iter[T]`](reference/iter.md)

## Error handling

- [`Result[T, E]`](reference/result.md)
- [`ResultType[T, E]`](reference/resulttype.md)
- [`Ok[T]`](reference/ok.md)
- [`Err[E]`](reference/err.md)
- [`ResultUnwrapError`](reference/resultunwraperror.md)

## Optional values

- [`Option[T]`](reference/option.md)
- [`Some[T]`](reference/some.md)
- [`NONE`](reference/null.md)
- [`OptionUnwrapError`](reference/optionunwraperror.md)
