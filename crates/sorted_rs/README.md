# sorted_rs

This crate aims to centralize as much as possible the logic related to sorted containers.

The main crate then handle the bridge between the "pure rust" logic and the Python interface.

## Informations about sorted set operations

Reimplementing the sorted set operations from `sortedcontainers` has proven to be a bit tricky, due to many confusing and poorly abstracted conditional paths in the original implementation, as well as the added fact that we had to:

1. Abstract and separate this crate from the pyclasses
2. Handle the `Mutex` locks across crates

Below is some doc resulting in my work trying to map clearly the original impl.

### Summary

If the op is mutating, returns `()` else `Self`.

| name                        | mutating | arg kind | Atomic op | Path  |
| --------------------------- | -------- | -------- | --------- | ----- |
| union                       | false    | Tuple    | -         | **1** |
| intersection                | false    | Tuple    | -         | **2** |
| difference                  | false    | Tuple    | -         | **2** |
| symmetric_difference        | false    | Any      | -         | **2** |
| intersection_update         | true     | Tuple    | -         | **3** |
| symmetric_difference_update | true     | Any      | -         | **3** |
| difference_update           | true     | Tuple    | `discard` | **4** |
| update                      | true     | Tuple    | `add`     | **4** |

### Execution paths in original sortedcontainers

#### 1. union -> update

1. `out = set(self.iter().chain(*iterables))` (set is effectively created once in `self.update()`)
2. `self._from_set(out)` -> `self.__init__(out)` -> `self.update(Arg::Set(out))` -> `concat_path::update(Arg::Set(out))`

#### 2. setop -> update

Same as `union`, but replaces step **1** with the corresponding set method.

#### 3. concat -> update

1. `self.set.method_correspondante(in)` (`in` is `Arg::Any` if sym_diff, `Arg::Tuple` if intersection)
2. `self.list.clear()`
3. `self.list.update(self.set)`

#### 4. concat -> len check -> update OR loop

1. `out = set(chain(*iterables))`
2. `if out_is_big {concat_path(Args::Set(out))} else {out.iter().for_each(method)}`

### Paths for the new impl (sketch)

`fn update_list(set)` is:

1. `self.list.clear()`
2. `self.list.update(set)`

We also drop (at least until running focused benchmarks) the `len` check, as it is not clear that it is worth the added complexity, since a rust `Vec` and a Python `list` clearly have different performance characteristics.

#### 1. apply transform

1. `out = self.set.method(in)`
2. `self.wrap(out)`

#### 2. concat -> update

1. `self.set.method_correspondante(in)`
2. `self.update_list(self.set)`

## Invariants

Note: Wrote by Claude clanker, not rechecked toroughly.

The sorted containers rely on the invariants below. Most are **not** encoded in the types: they are maintained by hand across `traits.rs`, `list.rs`, `keylist.rs` and `inner.rs`.

Notation: `values` holds the elements, `keys` holds the sort keys (for `List`, `keys` and `values` are the same storage), `maxes` holds the last key of each chunk, `load` is the load factor.

### Structural

#### 1. Lockstep

- Invariant: `maxes.len() == keys.len() == values.len()`.
- Used by: `Bisect::new`, `expand`, `push_val`, `index_to_loc`, `islice_bounds`.
- Checked in `debug.rs`: yes.

#### 2. Non-empty chunks

- Invariant: every chunk in `keys` and `values` has at least one element.
- Used by: `.last().unwrap()` in `traits.rs` (`expand`, `push_chunk`, `delete_many`), `list.rs` (`delete`), `keylist.rs` (`delete`), `inner.rs` (`loc_last`, `index_to_loc`), `ops.rs` (`Locate`).
- Checked in `debug.rs`: no. The checker only asserts `len >= lower_limit` for every chunk except the last.

#### 3. Max mirror

- Invariant: `maxes[i] == keys[i].last()` for every `i`.
- Maintained by hand in: `expand`, `delete`, `push_val`, `delete_on_len_sup_to_half_load`.
- Checked in `debug.rs`: yes.

#### 4. Key/value lockstep (KeyList only)

- Invariant: `keys[i].len() == values[i].len()` for every `i`.
- Used by: `expand`, `KeyList::split_values`.
- Checked in `debug.rs`: yes.

#### 5. Length total

- Invariant: `len == Σ values[i].len()`.
- Updated by hand in: `append`, `delete`.
- Checked in `debug.rs`: yes.

#### 6. Chunk size bounds

- Invariant: `len <= 2 * load` for every chunk, and `len >= load / 2` for every chunk except the last.
- Enforced by: the upper bound at insert time (`Expand::new`), the lower bound at delete time (`SubListState::new`).
- Checked in `debug.rs`: yes.

#### 7. Length tree (`lengths`)

- Invariant: either empty (not built), or consistent with `values`: `inner[0] == len`, `inner.len() == offset + values.len()`, and each internal node equals the sum of its children.
- Built lazily on the first index access past the first chunk (`set_pos_usize`, `into_index`). For a single chunk, `[len]` with `offset = 0`, as in the original.
- Relied on by: `expand_at`, `delete_at`.
- Checked in `debug.rs`: yes, only when built.

### Locations

#### 8. `Loc` validity

- Invariant: `pos < maxes.len()` and `idx <= keys[pos].len()`.
- `idx == len` is only valid as the end sentinel returned by `last_sublist` (`iter/builder.rs`), never as an element access.

#### 9. Bisect results

- Invariant: `Bisect::AtLoc(loc)` returns `loc.idx <= chunk.len()`, so `Vec::insert` at that index is in bounds.

### Preconditions

#### 10. Bounds normalization

- Invariant: `Bounds { start, stop }` satisfies `start < stop <= len` before reaching `Placement::new`, `ForwardSlice::new`, `set_pos_usize`, or `Search::index_by_*`.
- Established only by `try_normalize`, but `Bounds` has `pub` fields, so nothing prevents misuse.

#### 11. Non-empty list for index operations

- Invariant: `index` operations call `try_normalize` first.
- Relied on by: `Search::on_index` (`Bisect::Empty` is `unreachable!()` there), `ListsMethods::count` in `list.rs`.

#### 12. Monotonic order

- Invariant: `bisect_right(v) >= bisect_left(v)` and `min.pos <= max.pos` for every range.
- Without it, the arithmetic in `count` and `get_slice` underflows.

#### 13. Non-zero step

- Invariant: `ops::Slice::new` is never called with `Nb::Zero`. It is UB otherwise (`unreachable_unchecked`).
- Relies on `slice.indices` rejecting a zero step.

#### 14. Load factor

- `load >= 4` (`Load::test_limit`) comes from the original `sortedcontainers` test suite. `debug.rs` asserts it, but it has not been established that the implementation requires it.
- `reset` does not validate it. `load == 0` panics on the first split, and `update_chunked` desynchronizes `len`.
- `2 * load` and the cursor packing (`u32` for `pos` and `idx`) must not overflow.

### Known gaps

- Python exceptions raised by the key function or by `__lt__` inside sort closures are converted to panics by `.unwrap()` (`keylist.rs` `extend`, `list.rs` `extend`).
- `core.rs` uses `try_read().unwrap()` / `try_write().unwrap()`, which panics on lock contention.
- `iter/core.rs` marks `InnerRef` as `Send` and `Sync`, and releases the read lock before iterating. Mutating the list while an iterator is alive is a data race in Rust terms; only the GIL currently prevents it.
- `Updater::try_as_updatable` takes a read lock on the other list. Two threads doing `a.extend(b)` and `b.extend(a)` can deadlock.

## Statically known runtime checks

Note: Wrote by Claude clanker, not rechecked toroughly. Planned to be used as a TODO list.

Runtime checks whose outcome is already fixed by the type or by the call site. Each group below is a candidate for encoding, so the check can disappear. The sites are listed by file and function, with the line number at the time of writing.

### Witness `NonEmpty` chunk

A chunk in `keys` or `values` that is never empty. Encoding it removes the `last().unwrap()` calls.

| Location                                                                             | Code                                   | Note                            |
| ------------------------------------------------------------------------------------ | -------------------------------------- | ------------------------------- |
| `traits.rs`, l.[77](src/traits.rs#L77)                                               | `half.last().unwrap()`                 | in `expand`                     |
| `traits.rs`, l.[79](src/traits.rs#L79)                                               | `self.keys()[pos].last().unwrap()`     | in `expand`                     |
| `traits.rs`, l.[119](src/traits.rs#L119)                                             | `self.keys().loc_last(&loc)`           | [^1]                            |
| `traits.rs`, l.[156](src/traits.rs#L156)                                             | `chunk.last().unwrap()`                | in `push_chunk`                 |
| `list.rs`, l.[89](src/list.rs#L89)                                                   | `self.values[prev].last().unwrap()`    | in `delete`                     |
| `list.rs`, l.[93](src/list.rs#L93)                                                   | `self.keys().loc_last(&loc)`           | in `delete`                     |
| `keylist.rs`, l.[86](src/keylist.rs#L86)                                             | `left[prev].last().unwrap()`           | in `delete`                     |
| `keylist.rs`, l.[91](src/keylist.rs#L91)                                             | `self.keys().loc_last(&loc)`           | in `delete`                     |
| `inner.rs`, l.[80](src/inner.rs#L80)                                                 | `self[loc.pos].last().unwrap()`        | in `Values::loc_last`           |
| `inner.rs`, l.[118](src/inner.rs#L118)                                               | `self.values[pos].len() - 1_usize`     | in `index_to_loc`, `Last`       |
| `inner.rs`, l.[125](src/inner.rs#L125)                                               | `self.values[pos].len()`               | in `index_to_loc`, `InLastList` |
| `inner.rs`, l.[150](src/inner.rs#L150)                                               | `self.values[pos].len()`               | in `set_pos`                    |
| `ops.rs`, l.[194](src/ops.rs#L194)                                                   | `values.last().unwrap()`               | in `Locate::new`                |
| `search.rs`, l.[55](src/search.rs#L55)                                               | `ops::Bisect::Empty => unreachable!()` | [^2]                            |
| `lengths.rs`, l.[95](src/lengths.rs#L95), l.[98](src/lengths.rs#L98)                 | `tree.last().unwrap()`                 | [^3]                            |
| `iter/impls.rs`, l.[80](src/iter/impls.rs#L80)                                       | `(v.len() - 1) as u64`                 | [^4]                            |
| `iter/builder.rs`, l.[92](src/iter/builder.rs#L92)                                   | `values[pos].len()`                    | in `islice_bounds`              |
| `list_update.rs`, l.[111](src/list_update.rs#L111), l.[120](src/list_update.rs#L120) | `self.list.push_chunk(py, chunk)`      | in `update_chunked`             |
| `list_update.rs`, l.[108](src/list_update.rs#L108)                                   | `(!chunk.is_empty()).then_some(chunk)` | [^5]                            |

[^1]: in `delete_on_len_sup_to_half_load`, guarded by `LenSupToHalfLoad`

[^2]: in `Search::on_index`; the chunk is non-empty after `try_normalize`

[^3]: in `build`; never `None`, since `tree` starts with two rows

[^4]: in `inner_next_back`; `v` comes from `last()`

[^5]: in `update_chunked`; redundant once `load >= 1`

### Witness `NonEmpty` list

A list with at least one chunk (`maxes` and `values` non-empty). Encoding it removes the `unreachable!()` calls and the `len() - 1` on `maxes` and `values`.

| Location                                             | Code                                   | Note                            |
| ---------------------------------------------------- | -------------------------------------- | ------------------------------- |
| `list.rs`, l.[65](src/list.rs#L65)                   | `ops::Bisect::Empty => unreachable!()` | [^6]                            |
| `search.rs`, l.[55](src/search.rs#L55)               | `ops::Bisect::Empty => unreachable!()` | [^7]                            |
| `traits.rs`, l.[161](src/traits.rs#L161)             | `self.maxes.len() - 1`                 | [^8]                            |
| `keylist.rs`, l.[42](src/keylist.rs#L42)             | `self.maxes.len() - 1`                 | [^9]                            |
| `inner.rs`, l.[117](src/inner.rs#L117)               | `self.values.len() - 1`                | in `index_to_loc`, `Last`       |
| `inner.rs`, l.[124](src/inner.rs#L124)               | `self.values.len() - 1`                | in `index_to_loc`, `InLastList` |
| `inner.rs`, l.[149](src/inner.rs#L149)               | `self.values.len() - 1`                | in `set_pos`                    |
| `inner.rs`, l.[165](src/inner.rs#L165)               | `self.values[0].len()`                 | in `set_pos_usize`              |
| `ops.rs`, l.[191](src/ops.rs#L191)                   | `values[0].len()`                      | [^10]                           |
| `ops.rs`, l.[193](src/ops.rs#L193)                   | `values[0].len()`                      | [^11]                           |
| `slice.rs`, l.[116](src/slice.rs#L116)               | `list.values.len() - 1`                | in `fwd_extract`                |
| `iter/builder.rs`, l.[91](src/iter/builder.rs#L91)   | `list.values.len() - 1`                | in `islice_bounds`              |
| `iter/builder.rs`, l.[118](src/iter/builder.rs#L118) | `list.maxes.len() - 1`                 | [^12]                           |

[^6]: in `count`; reached only after `Bisect::AtLoc` on the left bound

[^7]: in `Search::on_index`; reached only after `try_normalize`

[^8]: in `push_val`; reached only from the `MaxesLen` branch

[^9]: in `append`; reached only from the `MaxesLen` branch

[^10]: in `Locate::new`; guarded by `length >= 2`

[^11]: in `Locate::new`; guarded by `length >= 2`

[^12]: in `last_sublist`; guarded by `is_empty` in `irange_bounds`

### Witness `Bounds` (`start < stop <= len`)

Built once by `try_normalize`, then passed to the consumers. Encoding it removes the subtractions and ordering assumptions in them.

| Location                                                       | Code                                                   | Note                          |
| -------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------- |
| `ops.rs`, l.[75](src/ops.rs#L75)                               | `indices.stop - indices.start`                         | in `ForwardSlice::new`        |
| `ops.rs`, l.[50](src/ops.rs#L50)                               | `debug_assert!(indices.stop <= length)`                | in `Slice::new`               |
| `ops.rs`, l.[211](src/ops.rs#L211) to l.[217](src/ops.rs#L217) | `index < bounds.start`                                 | [^13]                         |
| `slice.rs`, l.[46](src/slice.rs#L46)                           | `ops::ForwardSlice::new(self.indices, self.ref_len)`   | in `extract_from`, `Backward` |
| `slice.rs`, l.[106](src/slice.rs#L106)                         | `list.set_pos_usize(slf.start)?`                       | in `fwd_extract`              |
| `slice.rs`, l.[121](src/slice.rs#L121)                         | `list.set_pos_usize(slf.stop)?`                        | in `fwd_extract`              |
| `slice.rs`, l.[107](src/slice.rs#L107)                         | `Loc::with_idx(min.idx + (slf.stop - slf.start))`      | in `fwd_extract`              |
| `slice.rs`, l.[162](src/slice.rs#L162)                         | `list.set_pos_usize(idx)?`                             | in `del_all`                  |
| `slice.rs`, l.[144](src/slice.rs#L144)                         | `reset_slice((0, self.start), self.ref_len)`           | [^14]                         |
| `slice.rs`, l.[147](src/slice.rs#L147)                         | `reset_slice((self.stop, self.ref_len), self.ref_len)` | [^15]                         |
| `bounds.rs`, l.[54](src/bounds.rs#L54)                         | `Bounds::new(start, stop.cast_unsigned())`             | [^16]                         |

[^13]: in `Placement::new`; ordering comparisons

[^14]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^15]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^16]: in `try_normalize`; this is where the witness is created

### Witness `Load` (`load >= 1`, bounded)

A load factor validated at construction: `>= 1`, with `2 * load` and `load / 2` precomputed without overflow. Encoding it removes the implicit assumptions in the chunk size checks.

| Location                                           | Code                                                | Note                   |
| -------------------------------------------------- | --------------------------------------------------- | ---------------------- |
| `inner.rs`, l.[34](src/inner.rs#L34)               | `self.0 * 2`                                        | [^17]                  |
| `inner.rs`, l.[38](src/inner.rs#L38)               | `self.0 / 2`                                        | in `Load::lower_limit` |
| `ops.rs`, l.[124](src/ops.rs#L124)                 | `data.load.higher_limit()`                          | in `Expand::new`       |
| `ops.rs`, l.[161](src/ops.rs#L161)                 | `data.load.lower_limit()`                           | in `SubListState::new` |
| `traits.rs`, l.[77](src/traits.rs#L77)             | `half.last().unwrap()`                              | [^18]                  |
| `list_update.rs`, l.[103](src/list_update.rs#L103) | `let load = *self.list.load;`                       | [^19]                  |
| `list_update.rs`, l.[108](src/list_update.rs#L108) | `(!chunk.is_empty()).then_some(chunk)`              | in `update_chunked`    |
| `iter/cursor.rs`, l.[91](src/iter/cursor.rs#L91)   | `(loc.pos as Self) << u32::BITS \| loc.idx as Self` | [^20]                  |

[^17]: in `Load::higher_limit`; `2 * load` overflow

[^18]: in `expand`; non-empty only if `load >= 1`

[^19]: in `update_chunked`; `take(load)` yields non-empty chunks only if `load >= 1`

[^20]: in `From<Loc> for u64`; `idx < 2^32` only if chunk length is bounded by `2 * load`

### Witness `NonZeroIsize` (step)

| Location                             | Code                                             | Note  |
| ------------------------------------ | ------------------------------------------------ | ----- |
| `ops.rs`, l.[53](src/ops.rs#L53)     | `Nb::Zero => unsafe { unreachable_unchecked() }` | [^21] |
| `slice.rs`, l.[25](src/slice.rs#L25) | `let step = i.step.into();`                      | [^22] |

[^21]: in `Slice::new`; removing the variant removes the `unsafe`

[^22]: in `SliceIdxs::new`; the place where the witness is created

### Witness `Split` (`MustSplit` guard)

| Location                                   | Code                                   | Note  |
| ------------------------------------------ | -------------------------------------- | ----- |
| `traits.rs`, l.[76](src/traits.rs#L76)     | `self.keys_mut()[pos].split_off(load)` | [^23] |
| `keylist.rs`, l.[123](src/keylist.rs#L123) | `self.values[pos].split_off(load)`     | [^24] |

[^23]: in `expand`; guarded by `keys()[pos].len() > higher_limit()`

[^24]: in `split_values`; same guard, shared with `keys`

### `Option` of the key function

| Location                                                                           | Code                                      | Note                |
| ---------------------------------------------------------------------------------- | ----------------------------------------- | ------------------- |
| `list_update.rs`, l.[42](src/list_update.rs#L42)                                   | `py_func: Option<&'py Bound<'py, PyAny>>` | [^25]               |
| `list_update.rs`, l.[70](src/list_update.rs#L70)                                   | `.sorted(false, self.py_func)?`           | in `merge_with`     |
| `list_update.rs`, l.[86](src/list_update.rs#L86) to l.[87](src/list_update.rs#L87) | `self.py_func.map_or_else(...)`           | in `strategy`       |
| `list_update.rs`, l.[110](src/list_update.rs#L110)                                 | `match self.py_func`                      | in `update_chunked` |

[^25]: in `Updater`; `List` always passes `None`, `KeyList` always passes `Some`

### Delete path witness

| Location                                 | Code                      | Note  |
| ---------------------------------------- | ------------------------- | ----- |
| `traits.rs`, l.[128](src/traits.rs#L128) | `let prev = loc.pos - 1;` | [^26] |

[^26]: in `delete_many`; guarded by `ManySublists`, which implies at least two chunks

## Other classified sites

Sites that are not statically encodable as above. They are kept here so the inventory is complete.

### I: internal invariant

Depends on the structure (lockstep, Loc validity, sorted order), not on a type that could hold it.

- `traits.rs`: `expand` (`keys_mut()[pos]`, inserts at `pos + 1`), `delete_many` (`values[prev]`), `Bisect::new` (`keys()[pos]`), `Expand::new` and `SubListState::new` (`keys()[pos]`).
- `list.rs`: `append` (`keys_mut()[loc.pos].insert`), `count` (`len() - into_index`, `right.idx - left.idx`, sorted order), `delete` (`len -= 1`), `find` (`values.loc`).
- `keylist.rs`: `append` (`values[pos]`, `loc_insert`), `delete` (`len -= 1`, `split_at_mut`, `left[prev]`, `right[0]`, `keys.remove`), `split_values` (`values.insert`).
- `inner.rs`: `Values::loc`, `loc_insert`, `loc_remove`, `loc_len`, `remove_at`.
- `ops.rs`: `ForwardSlice::new` (`8 * ...`), `Update::new` (`4 * len`).
- `bounds.rs`: `try_normalize` (`len().cast_signed()`).
- `search.rs`: `found_in_keys`, `found_in_values`, `increment`.
- `list_update.rs`: `update_chunked` (`len = it.len()`).
- `bisect.rs`: `resolve` (`vec[mid]`).
- `lengths.rs`: `delete_at`, `expand_at`, `loc_at` (tree indices), `pos_at` (`pos - offset`).
- `slice.rs`: `fwd_extract` (`start_list[min.idx..max.idx]`), `get_slice` (range bounds), `get_all` (`cast_signed`).
- `iter/builder.rs`: `last_sublist` (`keys()[pos].len()`).
- `iter/impls.rs`: `inner_next` (`+ 1`), `inner_next_back` (`at - 1`).
- `iter/cursor.rs`: `load`, `new_pos`, `From<Loc> for u64` (packing).

### R: runtime (user data, external calls, cache state)

- `traits.rs`: `expand` (`lengths` clear and `expand_at`), `reset` (`load.into()`).
- `list.rs`: `extend` (`compare_fast(...).unwrap()`, see Known gaps).
- `keylist.rs`: `extend` (`call1(...).unwrap()`, see Known gaps).
- `inner.rs`: `set_pos` (`last_len + index`, `index += len`), `set_pos_usize` (`index >= len`, `lengths.build`, `pos_at`).
- `ops.rs`: `Bisect::new` (`is_empty`, `pos == len`), `SubListState::new` (`keys().len() > 1`), `Update::new` (`is_empty`), `Locate::new` (`-last < i`).
- `bounds.rs`: `try_normalize` (user inputs), `Loc::into_index` (cache).
- `search.rs`: `on_index` (`MaxesLen`), `index_by_val`, `index_by_key` (bisect and `Placement` results).
- `list_update.rs`: `merge_with` (`sorted(...)?`), `try_as_updatable` (`cast_exact`), `update_chunked` (`collect::<PyResult>`), `ListAdd::Sorted` (read lock).
- `bisect.rs`: `resolve` (`func(...)?`, `__lt__`).
- `lengths.rs`: `pos_at` (`index -= index_child`), `build` (row shape checks).
- `iter/impls.rs`: `inner_next` and `inner_next_back` (`Option` guards).
- `iter/core.rs`: `from_owner` (read lock).
- `core.rs`: `try_read().unwrap()`, `try_write().unwrap()` (lock contention).

### U: unsafe, FFI, external contract

- `bounds.rs`: `From<PySliceIndices>` (`cast_unsigned` relies on the `slice.indices` contract).
- `iter/impls.rs`: `yield_ptr` (`Py_INCREF`).
- `iter/core.rs`: `Deref` (`&*self.list`), `unsafe impl Send` and `Sync`.
- `iter/traits.rs`: `unwrap_unchecked` and `cast_unchecked` (FFI contract).
