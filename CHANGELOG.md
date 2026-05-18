# Changelog

## [Unreleased]

### 🔄 Refactors

- **Internal**: Renamed `PyOk` and `PyErr` to `PyoOk` and `PyoErr` to avoid any confusion with `Pyo3` types.

### 📖 Documentation

- Various reorganisation changes to reduce redundancy
- Updated CONTRIBUTING.md with latest architecture status

### 🛠️ Other improvements

- Various internal improvements to scripts for code readability
- Added checks in `scripts/check_doscstrings` to flag invalid existing links in zensical navigation.

### 🔗 Dependencies

- Added [`sdsort`](https://github.com/eirikurt/sdsort) as a dev dependency for code formatting and sorting.

## [0.18.0] - 2026-05-18

### 💥 Breaking changes

- **Removed**: `PyoIterator::is_distinct` was doing the exact same thing as `PyoIterator::all_unique` without a key function.
- **Removed**: `Iter::{partition_all, partition_by}`. Call the corresponding functions in `cytoolz.itertoolz` instead.
- **Removed**: `Iter::unique_to_each`. It was acting like it was lazy, but it was immediately consuming the whole iterator and storing the results in memory, as well as being niche. If you need this behavior, use the `more_itertool` corresponding function instead.
- **Removed**: `PyoIterator::elements` -> this was not lazy, but the API acted like it was. If you need this behavior, call `collections::Counter::elements` instead.
- **Removed**: `Vec::most_common` -> simple wrapper around `collections::Counter::most_common`. Call it directly instead if you need it.
- **Removed**: `Iter::{split_before, split_after, split_at, split_into, split_when}`. Call the corresponding functions in `more_itertools` instead.
- **Behavior change**: `PyoIterator::partition` now return a tuple of `Vec` from a predicate function, instead of an `Iter` of `tuples` of length n. It aligns with the Rust implementation, and is often more useful in practice. Call the corresponding function in `cytoolz.itertoolz` instead if you need the old behavior.
- **API change**: `PyoIterator::all_unique` does not accept a key function anymore. If you need one, call `all_unique_by` instead.

### ✨ Enhancements

- **Check safety**: Guarantee singleton behavior for `Null`. Calling `Null()` will always return the same instance, which is `NONE`. This allows for identity checks (`is`) to work as expected with `Null`, and ensures that there are no multiple instances of `Null` floating around in the system. This also means that you can use `Null()` instead of `NONE` if you prefer, without worrying about breaking the singleton property.
- **Feat**: `PyoIterator::all_unique` is migrated to `PyoIterable`, meaning ALL collections now can call it without converting to an iterator first. `PyoSequence` and `PyoSet` (and by extension `Seq`, `Set`, etc...) have their own optimized implementations
- **Feat**: Added `PyoIterable::all_unique_by` for checking uniqueness based on a custom key function. This is the same as former `all_unique(key=...)`, but with a clearer name and intent.

### 🚀 Performance improvements

- Using `cast_exact/is_exact_instance_of` instead of `cast/is_instance_of` when interacting with pyochain types (Result/Option) in Rust methods bring an overall **+2% to +5%** performance gain across benchmarks. Various `Option/Result` methods, as well a `Iter::{try_reduce, try_find, try_fold}`, benefit from this change.
- `all_unique` now call cython level code and is has fast as the deprecated `is_distinct` method.
- `PyoIterable::all_unique_by` (old `PyoIterator::all_unique(key=...)` path) is now in Rust, and is 30 (100 elements) to 70% (500 elements) faster than the old implementation. Larger collections should see even better improvements.
- `PyoIterable::all_unique` is in Rust as well, with a slight performance **regression** of around 1 to 5% compared to the old `is_distinct` method. Note that if you were using `all_equal` instead, you should expect performance **improvement** instead, in the same ballpark as `all_unique_by`.

### 🔄 Refactors

- Renamed `PyNone` to `PyNull` in Rust to avoid confusion with Pyo3 types.
- Various cosmetic changes in Rust to improve readability and documentation
- Extracted `PyoIterable::{first, second}`, `PyoIterator::{chain, insert, step_by}`, `Iter::tail` from cytoolz to python, as they were trivial function calls who didn't needed any external dependency work.
- Extracted `PyoIterable::{last, length}` from cytoolz to Rust. No performance regression.

### 🐞 Bug fixes

- Replaced `cytoolz` calls in `Iter::skip`. Slight performance regression, but `cytoolz` wasn't creating a new Iterator, but was eagerly consuming the original one until n elements were skipped, which could be a problem with large n/collections, and could cause issues if lazyness was expected. The new implementation uses `itertools.islice`, which creates a new iterator that skips the first n elements without consuming the original one.

### 🛠️ Other improvements

- Benchmarks -> deletions, renaming and new ones
- Cleaned up unneded/redundants tests for args concatenation and Result
- Docstring checker was broken and not flagging uncorrect docstrings. Fixed.

### 📖 Documentation

- Small improvements in `Iter` and `PyoIterable` docstrings

## [0.17.0] - 2026-05-17

### 🏆 Highlights

#### Option redesign for 100% type safety

The `Option` type has been redesigned to be more consistent with `Result`, with a new "dummy" Protocol `OptionType` as the base class for `Some` and `NONE`.

This change allows pattern matching exhaustiveness checks and much simpler internal Rust implementation, since now `Option` is, just like `Result`, a type union from the POV of type checkers. In Rust, it's an empty struct. No performance impact has been observed.

### 💥 Breaking changes

#### Option redesign

The following methods have been removed or renamed:

- `Option::__init__` => `option` (pure function)
- `Option::if_true` => `then_if_true` (pure function)
- `Option::if_some` => `then_if_some` (pure function)

### 📖 Documentation

- New template sections in [CONTRIBUTING.md](CONTRIBUTING.md) for standardizing the CHANGELOG and releases

### 🔄 Refactors

- Caller sites of `Option::__init__` have been updated

### 🛠️ Other improvements

- Various new benchmarks to cover impacted methods from the breaking changes

## [0.16.0] - 2026-05-17

### ✨ Enhancements

- `Iter::for_each_star` now handle `args` and `kwargs` in the same way as `Iter::for_each`, allowing to pass arguments to the function being called for each item.

### 🚀 Performance improvements

- Migrated `Iter::for_each, for_each_star` to Rust. 1.5x-2x faster in average.
- `Range` did not have `__slots__` properly set. Fixed.

### 🔄 Refactors

- New traits in `rust::types` to improve the readability when handling Callable arguments with args and kwargs. instead of call(func, self, args, kwargs), we can now do func.call(self, args, kwargs).
- Internal scripts cosmetic changes to standardize them with recommended pattern -> `from pyochain import x` instead of `import pyochain as pc`

### 📖 Documentation

- Improvements to `traits::PyoSet` methods documentation, with better examples and explanations.
- Updated documentation to reflect the new pattern -> `from pyochain import x` instead of `import pyochain as pc`
- Various other minor improvements and fixes.

### 🛠️ Other improvements

- `pytest-benchmark` setup, used and now part of the developpement process to catch performance regressions.
- Various other minor improvements and fixes.
