# Types Overview

Pyochain provides a comprehensive set of abstract base classes and concrete collection types, inspired by Python's `collections.abc` module and Rust's iterator patterns.

## Core Traits

Pyochain's type system is built on two categories of traits:

1. **Fluent mixins** — providing methods for fluent method chaining and conversion to `Option` and `Result` types
2. **Abstract collection protocols** — mirroring Python's `collections.abc` hierarchy to ensure protocol compatibility

For concrete examples of these traits in action, see the [interoperability section](interoperability.md).

### Fluent mixins

Fluent mixins are standalone classes that can be subclassed to enable functional method composition.

They depend only on `Self` for their implementation, making them universally applicable.

| Trait | Purpose | Main Capabilities | Inherited by |
| --- | --- | --- | --- |
| `Pipeable` | Functional chaining | `into()`, `inspect()` for fluent method composition | All pyochain objects |
| `Checkable` | Conditional operations | `then()`, `ok_or()`, `err_or()` for wrapping in `Option`/`Result` | All pyochain iterables + `Option` + `Result` |

### Abstract Collection protocols

Abstract collection protocols form a hierarchy that mirrors Python's `collections.abc` module.

Each protocol extends the corresponding one from `collections.abc`, inheriting its interface while adding pyochain-specific functionality.

Concrete types must implement the required methods (dunders) to satisfy the protocol contract.

| Trait | Extends | Inherits from `collections.abc` | Required Methods (for concrete types) | Concrete Types |
| --- | --- | --- | --- | --- |
| `PyoIterable[T]` | `Pipeable`, `Checkable` | `Iterable[T]` | `__iter__` | All pyochain collections |
| `PyoIterator[T]` | `PyoIterable[T]` | `Iterator[T]` | `__iter__`, `__next__` | `Iter[T]` |
| `PyoCollection[T]` | `PyoIterable[T]` | `Collection[T]` | `__contains__`, `__iter__`, `__len__` | All eager collections |
| `PyoSequence[T]` | `PyoCollection[T]` | `Sequence[T]` | `__getitem__`, `__len__` | `Seq[T]` |
| `PyoMutableSequence[T]` | `PyoSequence[T]` | `MutableSequence[T]` | `__setitem__`, `__delitem__`, `insert` | `Vec[T]` |
| `PyoSet[T]` | `PyoCollection[T]` | `Set[T]` | `__contains__`, `__iter__`, `__len__` | `Set[T]`, `PyoKeysView[K]`, `PyoItemsView[K,V]` |
| `PyoMappingView[T]` | `PyoCollection[T]` | `MappingView` | `__len__` | All mapping views |
| `PyoMapping[K, V]` | `PyoCollection[K]` | `Mapping[K, V]` | `__getitem__`, `__iter__`, `__len__` | `Dict[K,V]` (read-only interface) |
| `PyoMutableMapping[K, V]` | `PyoMapping[K, V]` | `MutableMapping[K, V]` | `__setitem__`, `__delitem__` | `Dict[K,V]` |

## Concrete Collections & Iterators

Pyochain provides concrete collection types that implement the abstract protocols described above.

All collections can be created from any object implementing Python's `Iterable` protocol.

Since these types fully implement their corresponding `collections.abc` protocols , they can act as drop-in replacements for their Python standard library counterparts.

### Concrete Collection Types

| Type                | Underlying Structure | Implements `collections.abc`        | Ordered | Uniqueness | Mutability |
|---------------------|----------------------|-------------------------------------|---------|------------|------------|
| `Iter[T]`           | `Iterator[T]`        | `Iterator[T]`                       | N/A     | N/A        | N/A        |
| `Seq[T]`            | `tuple[T]`           | `Sequence[T]`                       | Yes     | No         | No         |
| `Vec[T]`            | `list[T]`            | `MutableSequence[T]`                | Yes     | No         | Yes        |
| `Set[T]`            | `frozenset[T]`       | `Set[T]`                            | No      | Yes        | No         |
| `SetMut[T]`         | `set[T]`             | `MutableSet[T]`                     | No      | Yes        | Yes        |
| `Dict[K,V]`         | `dict[K, V]`         | `MutableMapping[K, V]`              | Yes     | Keys       | Yes        |
| `PyoKeysView[K]`    | `KeysView[K]`        | `KeysView[K]`, `Set[K]`             | No      | Yes        | No         |
| `PyoValuesView[V]`  | `ValuesView[V]`      | `ValuesView[V]`                     | No      | No         | No         |
| `PyoItemsView[K,V]` | `ItemsView[K,V]`     | `ItemsView[K,V]`, `Set[tuple[K,V]]` | No      | Yes        | No         |

## Option & Result Types

Pyochain provides two fundamental types for explicit handling of nullable values and errors, both implementing the fluent traits (`Pipeable` and `Checkable`).

### Option[T]

`Option[T]` represents values that may or may not be present, serving as a type-safe alternative to using `None`.

It provides methods to handle absence safely and fluently.

### Result[T, E]

`Result[T, E]` represents the outcome of operations that can succeed or fail, promoting explicit error handling without exceptions.

A `Result[T, E]` is either `Ok(value)` for successful operations, or `Err(error)` for failures.

Using `Result` as a return type clearly signals to callers that error handling is required.
