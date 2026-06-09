# Changelog

## [Unreleased]

### 🚀 Performance improvements

- `Iter::__bool__` is now **1.16x** faster by avoiding to use `itertools::islice`, this logic has been replaced by a `next` call and sentinel object check.

## [0.25.0] - 2026-06-09

### 🏆 Highlights

This release main change is a refactor of `Iter` and `PyoIterator` methods -> almost all methods from `Iter` have been moved to `PyoIterator`.

Only a handful of them remain in `Iter`, mostly instance methods relying on direct manipulation of the inner iterator, as well as the dunder overrides for concrete implemetnation.

### 💥 Breaking changes

- **API change**: `Iter::filter_star` does not accept arbitrary `Iterable` as elements anymore, only tuples.
- **API change**: The default `Seq` collector for `Iter::collect` has been deleted. All `.collect()` becomes `.collect(Seq)` if you want to keep the same behavior as before.

### 🚀 Performance improvements

- **no-copy**: Avoid unecessary copy in `Vec::repeat` by constructing the new instance by reference.
- `Iter::filter_star` has been moved to Rust. **1.45x** to **1.50x** faster across benchmarks (tuples with 2, 3, 4, 5 elements).
- `PyoIterator::{all, any}` with predicate function migrated to Rust.**1.02x** to **1.05x** faster.

### ✨ Enhancements

- **typing**: Improved the input types of various `PyoSet` methods to align them with typeshed, as they were too strict.
- **typing**: `Iter::{map_star, filter_star}` can handle 1 element tuples now.
- **typing**: `PyoIterable::iter` now returns a `PyoIterator` instead of an `Iter`, allowing more flexibility for custom overrides.

### 🛠️ Other improvements

- Splitted check_nav and generate_docs logic in two separate scripts. Closes issue #15.

## [0.24.0] - 2026-05-30

### 🏆 Highlights

#### `Iter::from_fn` in Rust

This release improve the performance of `Iter::from_fn` by:

- **1.35x** with NO `args`/`kwargs` (i.e `f() -> Option[T]`)
- **1.5x** with `args` OR `kwargs`
- **1.6x** with `args` AND `kwargs`

This means that you can define inline generators/iterators in a new way, without much, if any, performance trade-off in respect to the other three python-native paths.

In the benchmarks, only generator comprehensions has a slight performance advantage over `Iter::from_fn`.

This is fine, because comprehensions should be the go-to solution whenever possible, as they are concise, idiomatic/pythonic, and fast.

##### Performance vs python-native paths

The presence or absence of `*args` and `**kwargs` didn't have any substantial impact on the relative performance.

Python way                                           | Relative performance
-----------------------------------------------------|-----------
`Iterator` class with `__iter__` and `__next__`      | **Identical**
function/method with `yield` statements              | **Identical**
generator comprehension, i.e `(x for x in iterable)` | **0.9** to **0.95x** (i.e `from_fn` is 5-10% slower)

### 💥 Breaking changes

- **Removed**: `PyoMutableSequence::{into_iter, extend_move}`. What led to this decision is the fact that Python nature made these methods very situational, as well as their poor speed. `into_iter` can be replaced by `iter`, and `x.extend_move(y)` by `x.extend(y)` followed by `y.clear()`.

### 🚀 Performance improvements

- `PyoMutableSequence::truncate` is now **11x** faster. The old implementation used an inefficient loop with `pop()` calls, while the new one uses `del self[length:]`, after double-checking that this is a no-copy operation.
- `PyoMutableSequence::drain` logic has been improved to avoid `pop` calls, which made it **6.5x** faster. The subsequent Rust migration upped the improvement to **10.69x**.
- `PyoMutableSequence::extract_if` is the same story as `drain`. **5.18x** improvement from a logic change, upped to **8.28x** once ported to Rust.
- `Iter::successors` migrated to Rust. **1.3x** faster.

### 🔗 Dependencies

- **Dev**: Deleted unused script dependencies: `typer`, `polars`.

### 🛠️ Other improvements

- **Refactor**: Minor readability improvements in `scripts/`.

## [0.23.1] - 2026-28-05

### 🐞 Bug fixes

- **typing**: `PyoIterator::sum` has been giving false positives for a while. This is now fixed, with the same inputs/outputs as the python builtin.

### 🛠️ Other improvements

- **Internal**: Python dev version is by default 3.13 to avoid any unsupported patterns on our minimum supported version.
- **Imports**: Prioritizing `typing::TYPE_CHECKING` blocks import whenever possible. This could *maybe* speed-up initial import time.

## [0.23.0] - 2026-28-05

### 🏆 Highlights

- **Feat**: `collections`, new module mimicking python `collections` module, aiming to provide additional, more specialized collections types.

- **Feat**: `collections::StableSet`, a mutable collection of unique elements which remember their insertion order. Uses `dict::fromkeys` internally, thus has the same characteristics as a `dict` regarding lookup/insertion/deletion/iteration performance.
- **Feat**: `collections::Deque`, pyochain version of python builtin `collections::deque`.
- **Feat**: `SliceView`, a zero-copy, composable slice view over any `collections.abc.Sequence`. It allows to create views into existing sequences without copying the data, and to compose them together in O(1) time. It also has an `advance` method for shifting the view's window forward or backward in-place, which can be useful for sliding windows. Credits to [@eirikurt](https://github.com/julianofischer/sliceview) for the implementation and the idea, as well as[@hwelch-fle](https://github.com/hwelch-fle/sliceview) for the typing improvements work, which I used as base for the integration in pyochain.

### 🆕 New features

- `Iter::map_with` for mapping multiple iterables at once, just like `map` builtin when provided with multiple iterables.
- Improved `__eq__` for `Set`, `SetMut`, and `Dict`, with similar behavior than their internal data structures, as well as pyochain objects handling.
- `Set::__hash__`, with similar behavior than it's internal `frozenset` structure
- `repeat` method for `Seq` and `Vec`, which call the `__mul__` dunder method.
- `Vec::repeat_mut`, equivalent to `my_list *= n`.
- `Iter::from_repeat` for repeating an object n times as elements of an `Iter`.
- `Dict::union_mut`, similar to `dic_1 |= dic_2` for in-place union of two dicts.
- `Vec::into_iter` for consuming a `Vec` into an `Iter` that yield and pop each element from the `Vec`.
- `abc::{Into, Inspect}` mixins, each providing one method from the original `Pipeable`. The latter inherit from both of them, thus no behavior change.
- `abc::{PyoMutableSet, PyoSized, PyoReversible, PyoContainer}` ABCs. This don't impact runtime nor existing API, but is used for the ABC hierarchy tree to better mimick python `collections.abc`.
- `PyoSet::is_superset_strict` was missing. Added.
- `Dict::from_keys`, equivalent to `dict.fromkeys` python builtin.

### 💥 Breaking changes

- **Vec MRO**: `Vec` does not inherit from `Seq` anymore, only from `PyoMutableSequence`. This means that if you were relying on `Vec` being a subclass of `Seq` for type checking or isinstance checks, you should now check against `PyoMutableSequence` instead.
- **API change**: `PyoIterator::{argmin, argmax}` have been renamed to `arg_min` and `arg_max`, and their `key` argument version is now in the form of a separate method, `arg_min_by` and `arg_max_by`. Just like the last releases with similar changes, adapt your code in consequence.
- **Removed**: `PyoCollection::repeat` has been removed. You can do `my_collection.into(Iter.from_repeat, n)` to get the exact same behavior.
- **API change**: `Dict::merge` has been renamed to `union`.
- **API change**: `PyoIterable::{all, any, join, sum, min, my_by, max, max_by, all_unique_by, unpack_into}` are moved to `PyoIterator`. Simply add a call to `iter()` in impacted code if a `PyoCollection` was used.
- **API change**: `PyoIterable::length` has been removed. `PyoIterator::count` and `PyoSized::len` are their replacement. Closer to Rust semantics, and do things more explicitely, as the former is a full iteration that count the elements, while the latter is a call to `len()`.
- **API change**: All the methods that have been moved from `PyoIterator` to `Iter` in the **0.20.0** release are now back in their original ABC. To handle this, a new `_from_iterable` private method has been added, the idea being identical to what python stdlib does with set ABCs. See the method documentation for more details.
- **API change**: `Iter::with_position` now yield `Position(StrEnum)`  values instead of literal strings.
- **Removed**: `PyoIterable::all_unique` has been removed and is now only on `PyoIterator`. If you used it on a `PyoCollection`, you can either add a call to `iter()` before, or if you want to keep the exact same underlying implementation, compare the length of the collection with the length of a `Set` created from it. Examples and explanations in the documentation of `all_unique` method.
- **Removed**: `PyoIterable::second`. Use 2 calls to `itertator.next()` or `sequence[1]` instead.

### 🚀 Performance improvements

- **Moved to 🦀** -> `Iter::filter_map`. *1.15x* (64 elements) to *1.25x* (256, 1024, 4096 elements) faster.
- **Moved to 🦀** -> `Iter::filter_map_star`. *1.24x* (64 elements) to *1.38x* (256, 1024, 4096 elements) faster.
- **Moved to 🦀** -> `Iter::scan`. More or less *1.35x* faster across sizes (64, 256, 1024, 4096, 16384 elements).
- **Moved to 🦀** -> `Iter::map_while`. *1.38x* to *1.44x* faster across sizes.
- **tail no-copy**: `PyoIterator::tail` created internally a `deque` who was then re-wrapped in a `Seq` for the return. It now directly return a pyochain `Deque` created by reference from the aforementionned `deque`.

### ✨ Enhancements

- **typing**: `Iter::{filter, filter_false}` now handle type narrowing for `T | None` and `TypeGuard` when possible.
- **API**: `Iter::{filter, filter_false}` provided closure is now optional, just like python builtins.
- **API**: `Iter::{partition, collect}` has been moved to `PyoIterator`, meaning all user-defined iterators can now call them.
- **API**: `Iter::from_fn` now accepts `*args` and `**kwargs` to be passed to the generator function. Inputs are matched to minimize overhead and keep original performance if no additional arguments are needed.

### 📖 Documentation

- Correctly handle hyperlinks in `See Also` sections. WIP to do the same for the rest of the sections.
- Various docstrings changes with better explanations and examples.

### 📦 Build system

- **Fix**: Added `maturin` to [pyproject.toml](pyproject.toml) as a dev dependency.

### 🛠️ Other improvements

- Added `--preview` option to `Ruff` format workflow to garantuee markdown formatting
- Reorganized the [`abc`](src/pyochain/abc) submodule by splitting the monolithic `_iterable.py` file into multiple ones with more specific content.
- Moveed the views objects to a dedicated file for readability.
- Improved the doc generator script redability and it's handling of submodules.

## [0.22.0] - 2026-05-21

### 💥 Breaking changes

- **API**: `Vec::sort` is split into `Vec::sort` and `Vec::sort_by`. If you were using `Vec::sort(key=...)`, you should now use `Vec::sort_by(key=...)` instead. This aligns it with `PyoIterator::{sort, sort_by}` change from last release.

### ✨ Enhancements

- **Feat**: Added a clearer repr for `Range` that shows the start, stop and step values.
- **Feat**: `Vec` have an `__eq__` method. expect same results than `list::__eq__`, except that it also works with other `Vec` instances.
- **Feat**: `Seq` also now have an `__eq__` method. Same story as `Vec`. It also has an `__hash__` method, which is the same as `tuple::__hash__`, and is compatible with it, meaning that a `Seq` and a `tuple` with the same content will have the same hash.

### 🔄 Refactors

- Less code duplication in scripts with a common module

### 🛠️ Other improvements

- New pytest plugin improving docstring/markdown testing
- Separated `Ruff` config in a dedicated toml file

### 📖 Documentation

- Various improvements in docstrings, regarding formatting for the website, additional examples, etc..

## [0.21.0] - 2026-05-21

### 💥 Breaking changes

- **Removed**: `Set::r_intersection, r_union, r_difference, r_symmetric_difference` -> these methods were redundant and not really useful in practice, as the same behavior can be achieved by swapping the order of the sets in the non-reverse methods. For example, `a.r_intersection(b)` is equivalent to `b.intersection(a)`, and so on for the other methods.
- **Migrated**: `PyoKeysView`, `PyoValuesView` and `PyoItemsView` are moved from `abc` submodule to surface-level API. If you previously imported them as `from pyochain.abc`, update your imports to `from pyochain` instead.

### 🐞 Bug fixes

- `PyoKeysView` and `PyoItemsView` lost their set-like methods `union`, `intersection`, etc... in the last release. It was an overlook of the removal of these methods from the `PyoSet` ABC. This is now fixed.

### ✨ Enhancements

- **Better returns**: Due to the aforementioned bugfix, I took the opportunity to double check what was really returned from those operations before being wrapped in a `self::__class__` call. Turns out that by default, keys and item views were returning `set` objects, which were immediately re-wrapped in their callers, which may have entailed unecessary copies, but most importantly the API loss of an already constructed object with more methods (`SetMut`). Now, each class (`PyoKeysView`, `PyoItemsView`, `Set`, `SetMut`) has a dedicated implementation of these methods, which garantee **no-copy** behavior with a better API.

## [0.20.0] - 2026-05-21

### 💥 Breaking changes

- **API change**: `pyochain::traits` module is now `pyochain::abc`. If you were importing from `pyochain.traits`, update your imports to `pyochain.abc` instead.
- **Removed**: `PyoIterable::new`. Call `__init__(())` for the same behavior, e.g `Seq(())`, `Iter(())`, etc...
- **Removed**: `Unzipped` and `Peekable` dataclasses. The `Iter` methods who constructed them now simply return tuples instead, simplifying the API and improving speed.
- **API change**: `Iter::sort` (now in base class `PyoIterator`) has been split into `Iter::sort` and `Iter::sort_by`. If you were using `Iter::sort(key=...)`, you should now use `Iter::sort_by(key=...)` instead. This should bring typing improvements as well as a clearer API.
- **API change**: `PyoIterable::__init__` is deleted. This means that subclasses are free to implement their own constructors, without typing constraints nor default behavior.

#### Methods migration to concrete parents

If you did not define custom classes from `PyoSet` or `PyoIterator`, skip to the *Enhancements* section.

---

The `__init__` deletion from `PyoIterable` also means that all ABCs methods that relied on `self::__class__` needed to move to their concrete pyochain parents, as they couldn't stay (nor should have ever been) purely abstract.

This concerns:

from `PyoSet` to `Set` ->

`intersection`, `r_intersection`, `union`, `r_union`, `difference`, `r_difference`, `symmetric_difference`, `r_symmetric_difference`

---

from `PyoIterator` to `Iter` ->

`take_while`, `skip_while`, `compress`, `unique`, `unique_by`, `take`, `skip`, `step_by`, `slice`, `cycle`, `insert`, `intersperse`, `chain`, `accumulate`

---

### ✨ Enhancements

- **Migrated**: `Iter::{collect_into, try_collect, sort, tail}` have been moved to `PyoIterator`, meaning all user-defined subclasses can now call them.
- **Migrated**: `Vec::{drain, extract_if}` have been moved to `PyoMutableSequence`, meaning all user-defined mutable sequence subclasses can now call them.

### 📖 Documentation

- Improved various classes and methods documentation.
- Reformatted all code examples with Ruff.
- Added dev documentation regarding design choices on where to implement a method (abstract vs concrete class).

### 🔄 Refactors

- Separated some classes in dedicated files to reduce the size of `_iter.py` and improve readability. `Seq` is now in `_seq.py`, `Set` and `SetMut` are now in `_set.py` notably.

## [0.19.0] - 2026-05-20

### 🏆 Highlights

- **dependency-free**: pyochain is now dependency free and do not need `cytoolz` anymore. All the methods that were using it have been reimplemented in Rust or removed. See the breaking changes and performance improvements/regressions sections for more details.

### 💥 Breaking changes

- **API change**: `Iter::try_for_each` now returns `Result[tuple[()], E]` instead of `Result[None, E]`.
- **API change**: `PyoIterator::unique` is split into `Iter::unique` and `Iter::unique_by`. If you were using `Iter::unique(key=...)`, you should now use `Iter::unique_by(key=...)` instead.

Just like last release, some methods have beeen removed for a leaner, simpler API and a quicker no-dependency migration path.

See the table below for the removed methods and their recommended alternatives if you need the same behavior:

Method name | Equivalent | Notes
--- | --- | ---
`Iter::diff_at` | `cytoolz::itertoolz::diff` | -
`Iter::is_strictly_n` | `more_itertools::strictly_n` | -
`Iter::top_n` | `cytoolz::itertoolz::topk` | -
`PyoIterator::random_sample` | `cytoolz::itertoolz::random_sample` | -
`PyoIterator::interleave` | `cytoolz::itertoolz::interleave` | Unpack `self` and the other iterables in a single one before calling the function.

### 🆕 New features

- `Vec::concat_mut` for in-place concatenation of another `Vec` or `list`
- `Seq::concat` for concatenation of another `Seq` or `tuple`.

### 🚀 Performance improvements

- `PyoMutableSequence::extend_move` doesn't use `functools::partial` internally anymore. Expect some very light performance improvements.

#### Rust migrations🦀

Various methods have been migrated to Rust, from Python or Cython.

See the table below for the performance (2x means 2 times faster, 0.5x means 2 times slower).

Unfortunately, 3 methods have seen a performance regression vs the old Cython implementation, see the next section for details.

Method name                             | From     | Improvement                       | Notes
  ------------------------------------- | -------- | --------------------------------- | ---
`Iter::try_for_each`                    | *Python* | **4.6 - 4.7x**                    | -
`Iter::try_collect`                     | *Python* | **2.3 - 3x**                      | -
`PyoMutableSequence::retain`            | *Python* | **1.35 - 1.4x**                   | -
`Iter::{map_windows, map_windows_star}` | *Cython* | n=32: **1.17x**, n=128: **1.40x** | Slower for smaller window sizes (0.81x for n=2, 0.93x for n=8).
`Iter::map_juxt`                        | *Cython* | **1.2x to 1.5x**                  | Slower for a single func (0.95x), but not useful in practice, use `Iter::map` instead

### ⚠️ Performance regressions

The 3 following methods have been migrated from Cython to Rust, and have unfortunately seen a performance regression compared to their old Cython counterparts.

However, they are still much faster than a pure Python implementation.

See the table below for the details.

Method name               | Vs Cython | VS Python | Notes
------------------------- | --------- | --------- | ----------
`PyoIterator::unique_by`  | **0.95**  | **2.31x** | Equivalent to old `PyoIterator::unique(key=...)`
`PyoIterator::unique`     | **0.75**  | **7.3X**  | -
`PyoIterator::intersperse`| **0.60**  | **4.95x** | -

### ✨ Enhancements

- **typing**: `Some(NONE)` is directly inferred as `Option[T]`, lowering the number of potential errors.
- **typing**: `Iter::try_collect` has received new overloads and should be inferred more accurately now.
- **typing**: `Iter::map_juxt` has received new overloads to precisely infer the return type up to 10 functions.

### 🐞 Bug fixes

- `PyoIterator::{last, length}` were not properly handling custom Iterators and StopIteration errors. Fixed.

### 🔄 Refactors

- Renamed `converters.rs` file in Rust to `mixins.rs` to better reflect its content and purpose.
- **Internal**: Renamed `PyOk` and `PyErr` to `PyoOk` and `PyoErr` to avoid any confusion with `PyO3` types.
- **Internal**: Various renaming and cosmetic changes in Rust for better readability and consistency.

### 📖 Documentation

- Various reorganisation changes to reduce redundancy
- Updated CONTRIBUTING.md with latest architecture status
- Improved docstrings of various classes and methods with better examples and explanations
- **Fix**: `PyoIterator::for_each_star` was wrongly typed as working with an `Iterator` containing any `Iterable` elements, instead of only `tuple`s.

### 🧪 Tests

- standardized iteration sizes across benchmarks over iterators/collections

### 🛠️ Other improvements

- Various internal improvements to scripts for code readability
- Added checks in `scripts/check_doscstrings` to flag invalid existing links in zensical navigation, README and CONTRIBUTING.md, to prevent future broken links from being merged.

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
