# Changelog

## [unreleased]

### Breaking Changes

- `Option` and `Option::{if_true, if_some}` haave been respectively changed to pure functions `option`, `then_if_true`, `then_if_some`.
- `Option` is now, just like `Result`, a type union, and a "false" Protocol serves as the base class for `Some` and `NONE`.

### Internal

- Various new benchmarks covering some impacted methods

## [0.16] - 2026-05-17

### Features

- `Iter::for_each_star` now handle `args` and `kwargs` in the same way as `Iter::for_each`, allowing to pass arguments to the function being called for each item.

### Performance

- Migrated `Iter::for_each, for_each_star` to Rust. 1.5x-2x faster in average.
- `Range` did not have `__slots__` properly set. Fixed.

### Refactor

- New traits in `rust::types` to improve the readability when handling Callable arguments with args and kwargs. instead of call(func, self, args, kwargs), we can now do func.call(self, args, kwargs).
- Internal scripts cosmetic changes to standardize them with recommended pattern -> `from pyochain import x` instead of `import pyochain as pc`

### API Documentation

- Improvements to `traits::PyoSet` methods documentation, with better examples and explanations.
- Updated documentation to reflect the new pattern -> `from pyochain import x` instead of `import pyochain as pc`
- Various other minor improvements and fixes.

### Internal

- `pytest-benchmark` setup, used and now part of the developpement process to catch performance regressions.
- Various other minor improvements and fixes.
