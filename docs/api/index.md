# API reference

## Abstract Collection protocols

Abstract collection protocols form a hierarchy that mirrors Python's `collections.abc` module.

They expose the same API as their Python counterparts (required dunders and provided default implementations), with the additional pyochain-specific methods.

To see how each ABC must be implemented and what it offers, please refer to the [related official Python documentation](https://docs.python.org/3/library/collections.abc.html#collections-abstract-base-classes).

Simply add `Pyo` as a prefix to the Python ABC name to get the corresponding pyochain ABC.

Note that in the current version at the time of writing this (**v.0.26.0**), they are not all implemented yet.

## Concrete types

### Collections

Pyochain provides concrete collection types that implement the abstract protocols described above.

All collections can be created from any object implementing Python's `Iterable` protocol.

Since these types fully implement their corresponding interface, they can act as drop-in replacements for their Python standard library counterparts.

| Type                                                    | Underlying Structure | Ordered | Uniqueness | Mutability |
| ------------------------------------------------------- | -------------------- | ------- | ---------- | ---------- |
| [`Seq[T]`][pyochain.core._seq.Seq]                      | `tuple[T, ...]`      | Yes     | No         | No         |
| [`Vec[T]`][pyochain.core._vec.Vec]                      | `list[T]`            | Yes     | No         | Yes        |
| [`Set[T]`][pyochain.core._set.Set]                      | `frozenset[T]`       | No      | Yes        | No         |
| [`SetMut[T]`][pyochain.core._set.SetMut]                | `set[T]`             | No      | Yes        | Yes        |
| [`Dict[K,V]`][pyochain.core._dict.Dict]                 | `dict[K, V]`         | Yes     | Keys       | Yes        |
| [`PyoKeysView[K]`][pyochain.abc._views.PyoKeysView]     | `KeysView[K]`        | No      | Yes        | No         |
| [`PyoValuesView[V]`][pyochain.abc._views.PyoValuesView] | `ValuesView[V]`      | No      | No         | No         |
| [`PyoItemsView[K,V]`][pyochain.abc._views.PyoItemsView] | `ItemsView[K,V]`     | No      | Yes        | No         |
| [`Range`][pyochain.core._range.Range]                   | `range`              | Yes     | No         | No         |

### Iterators

| Type                                               | Underlying Structure |
| -------------------------------------------------- | -------------------- |
| [`Iter[T]`][pyochain.core._iterators.Iter]         | `Iterator[T]`        |
| [`Peekable[T]`][pyochain.core._iterators.Peekable] | `Iterator[T]`        |
