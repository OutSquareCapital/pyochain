# Release Note - Pyochain v0.6.0

This release marks a major turning point in the library's philosophy. The goal is to refocus `pyochain` on its functional primitives (`Iter`, `Result`, `Option`) and move away from "Dataframe-like" features for dictionaries, which are better handled by dedicated libraries like Polars.

The code has been massively refactored to be flatter, more maintainable, and more performant.

## 🚨 Breaking Changes

### 🏗️ Internal Architecture Overhaul

The package structure has been flattened to simplify imports and maintenance.

- **Removed**: Sub-packages `_core/`, `_dict/`, `_iter/`, `_results/`.
- **New**: Everything is now contained in single modules at the root of pyochain:
  - _core.py (Base classes `Pipeable`, `CommonBase`)
  - _dict.py (`Dict` implementation)
  - _iter.py (Common methods and utilities)
  - _lazy.py (`Iter` implementation)
  - _eager.py (`Seq`, `Vec`, `Set` implementations)
  - _option.py &_result.py

### 🗑️ Massive Cleanup of `Dict` API

Many methods from `Dict` have been removed. The philosophy is now: *"If you need to do complex aggregations, pivots, or joins, use Polars. Pyochain focuses on functional manipulation of standard dictionaries."*

**Removed Methods:**

- **Grouping & Aggregation**: `group_by_value`, `group_by_key`, `group_by_key_agg`, `group_by_value_agg`, `group_by_transform`.
- **Structure Transformation**: `pivot`, `unpivot`, `flatten`, `to_arrays`, `to_records`, `invert`, `rearrange`.
- **Joins**: `inner_join`, `left_join`, `merge`, `merge_with`, `diff`, `intersect_keys`, `diff_keys`.
- **Manipulation**: `pluck`, `with_nested_key`, `drop_nones`.
- **Misc**: `reduce_by`, `count_by`, `frequencies`.

### 🔄 Renaming in `Dict`

For better consistency with the standard API and Rust:

- `iter_keys()` ➡️ `keys_iter()`
- `iter_values()` ➡️ `values_iter()`
- `iter_items()` ➡️ `iter()` (Returns an `Iter[tuple[K, V]]`)

### ⚡ Changes in `Iter`

- **`group_by()` is now Lazy**:
  - **Before**: Returned a dictionary or materialized structure.
  - **Now**: Returns an `Iter[tuple[Key, Iter[Val]]]`. This allows processing infinite or very large streams without loading everything into memory. The behavior is similar to `itertools.groupby` (requires prior sorting).
- **Removed** `frequencies()` (use `Counter` or Polars).

## ✨ New Features

### 🆕 New Class `Set[T]`

Introduction of an immutable, unordered collection for unique elements.

- Wraps `set` or `frozenset`.
- Implements chainable set operations: `union`, `intersection`, `difference`, `symmetric_difference`, `is_subset`, etc.

### ⚙️ Global Configuration

Added a configuration module _config.py and the `PyochainConfig` class.

- Allows configuring global behavior, notably the text representation (`__repr__`) of iterators (number of items displayed by default).

---

### 📝 Migration Summary

If you were using `pyochain` as a "mini-pandas" for dictionaries, you will need to migrate that code to **Polars** or reimplement the logic with simple `map`/`filter`. If you were using it for functional chaining on iterators, the update should be smooth, except for `group_by` which now requires explicit consumption of subgroups, and `Seq` which lost its set-like capabilities.

## Deleted methods because equivalent in polars

Given a Dict(data).methods*() (or iter_items() when it make sense) what's the more or less equivalent as DataFrame(data).methods*()?

This serves more as rough guidelines and hints rather than exact equivalence.

When it's more pyochain centric, the pyochain equivalent is given.

## Deleted

### Dict

`.group_by_value()` -> `DataFrame.group_by()`
`.group_by_key()` -> `DataFrame.group_by()`
`.group_by_key_agg()` -> `DataFrame.group_by()`
`.group_by_value_agg()` -> `DataFrame.group_by()`
`.pivot()` -> `DataFrame.unpivot(variable_name="x").unnest("value").unpivot(index="x").pivot("x", index="variable")`
`.unpivot()` -> `DataFrame.unpivot()`
`.flatten()` -> `DataFrame.unnest()`
`.pluck()` -> `Expr.struct.field()`
`.inner_join()` -> `DataFrame.inner_join()`
`.left_join()` -> `DataFrame.left_join()`
`.rename()` -> `DataFrame.rename()`
`.invert()` -> `DataFrame.unpivot().group_by("value").all()`
`.merge()` -> `DataFrame.with_columns()`
`.merge_with()` -> `DataFrame.vstack().{group_by()/sum()/etc...}`
`.with_nested_key()` -> `DataFrame.with_columns()`
`.diff()` -> Exact output be done with custom logic using pl.lit, pl.when, group_by, etc. however there's far better way to do this in Polars, or with pyochain using sets on tuples
`intersect_keys()` -> `DataFrame.join(how="inner")`
`.diff_keys()` -> `DataFrame.join(how="anti")`
`.to_arrays()` -> `DataFrame.unnest()`
`.rearrange()` -> `Expr.replace()`
`.drop_nones()` -> explicit schema + `{Expr, DataFrame}.drop_nulls()`

### Iter

`.frequencies()` -> `Expr.value_counts()`
`.group_by()` (dictoolz) -> `DataFrame.group_by()`. The new `Iter.group_by()` is fully lazy and use `itertools.groupby()`.
`.with_{keys()/values()}` -> `Iter.zip().into(lambda x: pc.Dict(dict(x)))`
`.reduce_by()` -> `DataFrame.group_by().agg()`
`.count_by()` -> `DataFrame.group_by().count()`
`.to_records()` -> `Expr.list + Expr.struct` expressions.
`.group_by_transform()` -> `DataFrame.group_by().agg()`
