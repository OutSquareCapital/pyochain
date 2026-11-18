# Migrating from sortedcontainers to pyochain

This document provides guidance on how to migrate from the `sortedcontainers` library to `pyochain`.

The main takeaways are:

- Use composition instead of heritance or monkey-patching
- Expect more conformance to python `collections::abc` protocols
- Massive performance improvements (see benchmarks)
- Full interop with the rest of pyochain library
- Full type-checking support
- Python 3.13 >= support. HOWEVER, older versions are not supported.

## General

- Classes are frozen, meaning that dynamically changing the class at runtime with things such as `__setattr__`, `__delattr__`, and `__dict__` will raise runtime errors.
- Heritance is **NOT** supported, all classes are marked as `typing::final`, and will raise runtime errors if you try to inherit from them.
- No heritance relationships between the "concrete" sorted classes: `SortedKeyList` is not a subclass of `SortedList`.
- No class herit from python built-in types: `SortedDict` is not a subclass of `dict`, and `SortedSet` is not a subclass of `set`.
- Base classes/protocols who are type-checking only construct, such a `BaseSortedCollection`, are provided for duck-typing, if needed, and also allow to avoid duplicating the documentation.
- All private methods and attributes (those starting with `_`) are not accessible from Python, and are implementation details living in rust.
- `load` is fully private and can not be accessed from Python.
- All classes are explicit: There's no implicit dynamic dispatch from a `SortedList` to a `SortedKeyList` if a `key` is provided. `SortedKeySet` and `SortedKeyDict` are also not "hidden".
- Method aliases, such as `bisect` for `bisect_left`, are not provided.
- Mutability and iteration edge cases are a bit different. Expect `pyochain::SortedList` to be closer to python `list` than `sortedcontainers::SortedList` in some cases. Tests at [tests\test_sorted] demonstrate the differences.

## SortedList/SortedKeyList

- `SortedKeyList` constructor arguments order is flipped: It takes the `key` argument first, and the `iterable` second. The `key` must be provided explicitly, instead of defaulting to an identity function.
- `add` is replaced by `append`, which does not raise an error at runtime.
- `insert` ignores the `index`, and forward the `value` to `append`.
- `update` is replaced by `extend`, which does not raise an error at runtime.
