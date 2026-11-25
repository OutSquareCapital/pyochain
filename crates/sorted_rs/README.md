# sorted_rs

This crate aims to centralize as much as possible the logic related to sorted containers.

The main crate then handle the bridge between the "pure rust" logic and the Python interface.

## Current perf improvements ideas

Non-exhaustive list of ideas to improve the performance of the current implementation. None is measured yet. Ordered by expected impact.

### Lookups: `add`, `contains`, `remove`, `discard`, `index`, `count`, `irange`

- Every comparison of `bisect` calls `__lt__` through Python, about 20 times per operation (one bisect over `maxes`, one over the sublist). The sort already has fast paths for ints, floats and latin strings (`KeysCheck`), the bisect has none.
  - Idea: store the type of the keys in `InnerList`. It is set by the first insertion, and dropped for good as soon as one item doesn't match. `bisect` then matches on it, like `SortBy::sort`.
  - It has to be kept up to date by `append`, the three `ExtendKind` and `insert_at`.
- `List::count` restarts its right bisect from the start of `maxes`, instead of from the left result.

### `SortedKeyList`: key function called again on known keys

`keys` already holds the key of every item, but the key function is called again for:

- `extend` (`Big`) and `concat`: `n + m` calls instead of `m`.
- `extend` from another `SortedKeyList`: its keys are not reused.
- `reset` and the deletion of a small slice, which rebuild the list from the values.

No benchmark uses a real key function yet.

### Slicing

Reading:

- `sl[a:b]` builds a `Vec` of clones, then a `PyList` that increments each reference a second time. The views and `del_from_set` do the same. A lazy iterator with an exact length would avoid the intermediate `Vec`.

Deleting:

- `delete_fwd_small` clones everything around the slice, clears the list and sorts it again.
- `del sl[a:b]` deletes item by item, and each merge of sublists resets `lengths`, which is rebuilt for the next one. A contiguous range could be cut at both ends, with the sublists in between dropped at once.

With a step, reading and deleting:

- Each item looks up its sublist again from the root of the `lengths` tree. A cursor moving by `step` would avoid it.

### Rebuilding the list

- `reset` clones every item, clears the list and sorts again, although the items are already sorted. Since `load` is not mutable from Python, the construction API should be rethought together with it.
- `x.extend(x)` sorts the whole list twice its size.

### Minor

- `Dict::copy` rebuilds the `PyDict` key by key. `PyDict.copy()` may do, if the internal order of `data` is never observable.
- `Big` clones the existing values, then `clear` releases them.
- `Small` expands and updates `len` for each item. The thresholds `MIN_ITEMS = 4` and `4 * m >= n` of `Strategy` are not proven to be the best ones.

## Informations about sorted set operations

Reimplementing the sorted set operations from `sortedcontainers` has proven to be a bit tricky, due to many confusing and poorly abstracted conditional paths in the original implementation, as well as the added fact that we had to:

1. Abstract and separate this crate from the pyclasses
2. Handle the `RwLock` locks across crates

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

The `Atomic op` and `Path` columns describe the original `sortedcontainers` implementation. The current implementation doesn't use them: see [Paths in the current impl](#paths-in-the-current-impl).

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

### Paths in the current impl

Every set operation goes through one of the two entry points of `PySetDataRef` ([`set.rs`](src/set.rs)): `op` for the non-mutating ones, `op_mut` for the mutating ones. The four original paths are merged into these two.

The `len` check of the original paths **4** is dropped (at least until running focused benchmarks), as it is not clear that it is worth the added complexity, since a rust `Vec` and a Python `list` clearly have different performance characteristics. `Dict::update` still has a similar heuristic (`10 * pairs.len() > inner.len()`).

#### 1. apply transform (`op`)

1. `out = self.set.method(in)` (`SetOp::call`)
2. `Set::wrap(out)`: the values of `out` are collected, then sorted into a new list.

#### 2. concat -> update (`op_mut`)

1. `self.set.method_correspondante(in)` (`SetOp::call_mut`)
2. `self.list.clear()`
3. the values of `self.set` are extended into the now empty list, which always takes the "empty list" path (full sort, then chunking).

The set is mutated before the list is rebuilt: see [Known gaps](#known-gaps).

## Invariants

Note: Wrote by Claude clanker, not rechecked toroughly.

The sorted containers rely on the invariants below. Most are **not** encoded in the types: they are maintained by hand across `traits.rs`, `list.rs`, `keylist.rs` and `inner.rs`.

Notation:

- `values` holds the elements
- `keys` holds the sort keys (for `List`, `keys` and `values` are the same storage)
- `maxes` holds the last key of each chunk
- `load` is the load factor

### Structural

#### 1. Lockstep

- Invariant: `maxes.len() == keys.len() == values.len()`.
- Used by: `Bisect::new`, `expand`, `push_val`, `index_to_loc`, `islice_bounds`.
- Checked in `debug.rs`: yes.

#### 2. Non-empty chunks

- Encoded by `SubList` ([`sublist.rs`](src/sublist.rs)).

#### 3. Max mirror

- Invariant: `maxes[i] == keys[i].last()` for every `i`.
- Maintained by hand in: `expand`, `delete`, `push_val`, `delete_on_len_sup_to_min`, `Chunk::push_into`, `Batch::extend_into` and the `Small` arm of `ExtendKind::extend_into`.
- Checked in `debug.rs`: yes.

#### 4. Key/value lockstep (KeyList only)

- Invariant: `keys[i].len() == values[i].len()` for every `i`.
- Used by: `expand`, `KeyList::split_values`.
- Checked in `debug.rs`: yes.

#### 5. Length total

- Invariant: `len == Σ values[i].len()`.
- Updated by hand in: `append`, `delete`, and the three arms of `ExtendKind::extend_into` (`Big` sets it, `Medium` and `Small` add to it).
- Checked in `debug.rs`: yes.

#### 6. Chunk size bounds

- Invariant: `len <= max_size` (`2 * load`) for every chunk, and `len >= min_size` (`load / 2`) for every chunk except the last.
- Enforced by: the upper bound at insert time (`Expand::new`, and the `while` loop of `Batch::extend_into`, which splits as many times as needed), the lower bound at delete time (`SubListState::new`). `Big` builds chunks of exactly `load` items, except the last.
- `load` is validated by `Load` ([`load.rs`](src/load.rs)).
- Checked in `debug.rs`: yes.

#### 7. Length tree (`lengths`)

- Invariant: either empty (not built), or consistent with `values`: `offset == values.len().next_power_of_two() - 1`, `inner[0] == len`, `inner.len() == offset + values.len()`, each internal node equals the sum of its children, and the nodes without children are 0.
- Built lazily on the first index access past the first chunk (`set_pos_usize`, `into_index`).
- Relied on by: `add_at` (`expand_at` is `add_at(pos, 1)`), `delete_at`, `loc_at`, `pos_at`.
- `Batch::extend_into` only calls `add_at` while the index is built. A split clears it, so the remaining runs skip it, and it is rebuilt on demand.
- Checked in `debug.rs`: yes, only when built.

### Locations

#### 8. `Loc` validity

- Invariant: `pos < maxes.len()` and `idx <= keys[pos].len()`.
- `idx == len` is only valid as an end sentinel (`last_sublist` and `islice_bounds` in `iter/builder.rs`) or as an insertion position (`Loc::next_sorted` in `sort/chunk.rs`), never as an element access.

#### 9. Bisect results

- Invariant: `Bisect::AtLoc(loc)` returns `loc.idx <= chunk.len()`, so `Vec::insert` at that index is in bounds.

### Preconditions

#### 10. Bounds normalization

- Invariant: `Bounds { start, stop }` satisfies `start < stop <= len` before reaching `Placement::new`, `ForwardSlice::new`, `set_pos_usize`, or `Search::index_by_*`.
- Established by `try_normalize` for `index` and `islice`. For slices, by `From<PySliceIndices> for Bounds` together with the `start >= stop => NoOp` guard of `Slice::new`. `reset_slice` builds a `Bounds` directly and relies on the same guard.
- `Bounds` has `pub` fields, so nothing prevents misuse.

#### 11. Non-empty list for index operations

- Invariant: `index` operations call `try_normalize` first.
- Relied on by: `Search::on_index` (`Bisect::Empty` is `unreachable!()` there).
- `ListsMethods::count` in `list.rs` has the same `unreachable!()` on the right bound, but doesn't go through `try_normalize`: it relies on the `Bisect::AtLoc` of the left bound, which implies a non-empty list.

#### 12. Monotonic order

- Invariant: `bisect_right(v) >= bisect_left(v)` and `min.pos <= max.pos` for every range.
- Without it, the arithmetic in `count` and `get_slice` underflows.

#### 13. Non-zero step

- Invariant: `ops::Slice::new` is never called with `Nb::Zero`. It is UB otherwise (`unreachable_unchecked`).
- Relies on `slice.indices` rejecting a zero step.

### Sorted batches

#### 14. Sorted input

- Invariant: the items of a `SortedVec` are in non-decreasing order of their key, and `Loc::next_sorted` is only called on a non-empty list.
- Established by: `SortedVec::from_iter` and `SortedVec::from_vec` (sort), and `IntoList::extend_into`, which only picks `Medium` and `Small` when the list isn't empty.
- Relied on by: `Loc::next_sorted` (compares from the previous location, never goes back), `Batch::locate`, the `Small` arm of `ExtendKind::extend_into`.

#### 15. Batch application

- Invariant: every location of a `Batch` is computed against the list before any insertion, then applied from the last sublist to the first, so a split never shifts a location that wasn't applied yet.
- `SubList::insert_many` (`unsafe`: `ptr::copy`, `ptr::write`, `set_len`) requires one location per item, with sorted indices, each `<= len`. Not covered by Rust unit tests: only exercised through `Batch` by the Python tests.

### Known gaps

- `core.rs` uses `try_read().unwrap()` / `try_write().unwrap()`, which panics on lock contention, and on any re-entrant call made while the lock is held.
- `IntoList::PyIter` is consumed lazily, inside `ListsMethods::extend`. The Python code of the iterable, of the key function and of `__lt__` therefore runs while the write lock is held.
- Some updates aren't atomic:
  - the `Small` arm of `ExtendKind::extend_into` compares and inserts item by item, so an exception leaves a partial update;
  - `reset`, `delete_fwd_small`, `Dict::update` and `PySetDataRef::op_mut` call `clear` and then re-sort, so an exception in the second step leaves the list empty (for `op_mut`, with the Python set already updated).
  - `Medium` locates every item before mutating anything, and `Big` sorts before `clear`: both are atomic.
- `iter/core.rs` marks `InnerRef` as `Send` and `Sync`, and releases the read lock before iterating. Mutating the list while an iterator is alive is a data race in Rust terms; only the GIL currently prevents it.
- The iterator cursor (`iter/cursor.rs`) packs `pos` in the 32 high bits of a `u64`. `idx` is bounded by `Load` (invariant 6), `pos` is not: the number of chunks depends on the total length. It overflows past `2^32` chunks, i.e. about 4 billion elements with `load = 1`, or about 2 trillion with the default load. Two fixes are possible, none applied yet:
  - a split of the bits that depends on `load`, at the cost of a variable shift on every `next`;
  - cursors specialized on a constant split, with one pyclass per split (generated with the existing macros). Same cost as today (constant shift), so no perf gain or loss. A strict guarantee for small loads needs about 61 bits for `pos`, so several splits.

## Statically known runtime checks

Note: Wrote by Claude clanker, not rechecked toroughly. Planned to be used as a TODO list.

Runtime checks whose outcome is already fixed by the type or by the call site. Each group below is a candidate for encoding, so the check can disappear. The sites are listed by file and function, with the line number at the time of writing.

### Witness `NonEmpty` list

A list with at least one chunk (`maxes` and `values` non-empty). Encoding it removes the `unreachable!()` calls and the `len() - 1` on `maxes` and `values`.

| Location                                             | Code                                   | Note                            |
| ---------------------------------------------------- | -------------------------------------- | ------------------------------- |
| `list.rs`, l.[74](src/list.rs#L74)                   | `ops::Bisect::Empty => unreachable!()` | [^6]                            |
| `search.rs`, l.[55](src/search.rs#L55)               | `ops::Bisect::Empty => unreachable!()` | [^7]                            |
| `traits.rs`, l.[144](src/traits.rs#L144)             | `self.maxes.len() - 1`                 | [^8]                            |
| `keylist.rs`, l.[57](src/keylist.rs#L57)             | `self.maxes.len() - 1`                 | [^9]                            |
| `sort/chunk.rs`, l.[133](src/sort/chunk.rs#L133)     | `list.maxes.len() - 1`                 | [^25]                           |
| `inner.rs`, l.[137](src/inner.rs#L137)               | `self.values.len() - 1`                | in `index_to_loc`, `Last`       |
| `inner.rs`, l.[144](src/inner.rs#L144)               | `self.values.len() - 1`                | in `index_to_loc`, `InLastList` |
| `inner.rs`, l.[169](src/inner.rs#L169)               | `self.values.len() - 1`                | in `set_pos`                    |
| `inner.rs`, l.[185](src/inner.rs#L185)               | `self.values[0].len()`                 | in `set_pos_usize`              |
| `ops.rs`, l.[189](src/ops.rs#L189)                   | `list.values[0].len()`                 | [^10]                           |
| `ops.rs`, l.[191](src/ops.rs#L191)                   | `list.values[0].len()`                 | [^11]                           |
| `slice.rs`, l.[118](src/slice.rs#L118)               | `list.values.len() - 1`                | in `fwd_extract`                |
| `iter/builder.rs`, l.[91](src/iter/builder.rs#L91)   | `list.values.len() - 1`                | in `islice_bounds`              |
| `iter/builder.rs`, l.[118](src/iter/builder.rs#L118) | `list.maxes.len() - 1`                 | [^12]                           |

[^6]: in `count`; reached only after `Bisect::AtLoc` on the left bound

[^7]: in `Search::on_index`; reached only after `try_normalize`

[^8]: in `push_val`; reached only from the `MaxesLen` branch

[^9]: in `append`; reached only from the `MaxesLen` branch

[^10]: in `Locate::new`; guarded by `length >= 2`

[^11]: in `Locate::new`; guarded by `length >= 2`

[^12]: in `last_sublist`; guarded by `is_empty` in `irange_bounds`

[^25]: in `Loc::next_sorted`; only reached from `Medium` and `Small`, which require a non-empty list

### Witness `Bounds` (`start < stop <= len`)

Built once by `try_normalize` (or, for slices, by `From<PySliceIndices> for Bounds`), then passed to the consumers. Encoding it removes the subtractions and ordering assumptions in them.

| Location                                                            | Code                                                   | Note                          |
| ------------------------------------------------------------------- | ------------------------------------------------------ | ----------------------------- |
| `ops.rs`, l.[75](src/ops.rs#L75)                                    | `indices.stop - indices.start`                         | in `ForwardSlice::new`        |
| `ops.rs`, l.[50](src/ops.rs#L50)                                    | `debug_assert!(indices.stop <= length)`                | in `Slice::new`               |
| `ops.rs`, l.[209](src/ops.rs#L209) to l.[211](src/ops.rs#L211)      | `index < bounds.start`                                 | [^13]                         |
| `slice.rs`, l.[48](src/slice.rs#L48)                                | `ops::ForwardSlice::new(self.indices, self.ref_len)`   | in `extract_from`, `Backward` |
| `slice.rs`, l.[108](src/slice.rs#L108)                              | `list.set_pos_usize(slf.start)?`                       | in `fwd_extract`              |
| `slice.rs`, l.[123](src/slice.rs#L123)                              | `list.set_pos_usize(slf.stop)?`                        | in `fwd_extract`              |
| `slice.rs`, l.[109](src/slice.rs#L109)                              | `Loc::with_idx(min.idx + (slf.stop - slf.start))`      | in `fwd_extract`              |
| `slice.rs`, l.[164](src/slice.rs#L164)                              | `list.set_pos_usize(idx)`                              | in `del_all`                  |
| `slice.rs`, l.[146](src/slice.rs#L146)                              | `reset_slice((0, self.start), self.ref_len)`           | [^14]                         |
| `slice.rs`, l.[149](src/slice.rs#L149)                              | `reset_slice((self.stop, self.ref_len), self.ref_len)` | [^15]                         |
| `bounds.rs`, l.[19](src/bounds.rs#L19) to l.[23](src/bounds.rs#L23) | `if i.step > 0 { .. } else { .. }`                     | [^27]                         |
| `bounds.rs`, l.[54](src/bounds.rs#L54)                              | `Bounds::new(start, stop.cast_unsigned())`             | [^16]                         |

[^13]: in `Placement::new`; ordering comparisons

[^14]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^15]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^16]: in `try_normalize`; this is where the witness is created for `index` and `islice`

[^27]: in `From<PySliceIndices> for Bounds`; this is where the witness is created for slices, with the `start >= stop => NoOp` guard of `Slice::new` covering the empty case

### Witness `NonZeroIsize` (step)

| Location                             | Code                                             | Note  |
| ------------------------------------ | ------------------------------------------------ | ----- |
| `ops.rs`, l.[53](src/ops.rs#L53)     | `Nb::Zero => unsafe { unreachable_unchecked() }` | [^21] |
| `slice.rs`, l.[27](src/slice.rs#L27) | `let step = i.step.into();`                      | [^22] |

[^21]: in `Slice::new`; removing the variant removes the `unsafe`

[^22]: in `SliceIdxs::new`; the place where the witness is created

### Witness `Split` (`MustSplit` guard)

| Location                                   | Code                                   | Note  |
| ------------------------------------------ | -------------------------------------- | ----- |
| `traits.rs`, l.[65](src/traits.rs#L65)     | `self.keys_mut()[pos].split_off(load)` | [^23] |
| `keylist.rs`, l.[126](src/keylist.rs#L126) | `self.values[pos].split_off(load)`     | [^24] |

[^23]: in `expand`; guarded by `keys()[pos].len() > max_size()`

[^24]: in `split_values`; same guard, shared with `keys`

### Delete path witness

| Location                             | Code          | Note  |
| ------------------------------------ | ------------- | ----- |
| `inner.rs`, l.[78](src/inner.rs#L78) | `loc.pos - 1` | [^26] |

[^26]: in `Values::remove_merge`; guarded by `Merge::IntoPrev`, which `SubListState::new` only picks when `pos > 0`

## Other classified sites

Sites that are not statically encodable as above. They are kept here so the inventory is complete.

### I: internal invariant

Depends on the structure (lockstep, Loc validity, sorted order), not on a type that could hold it.

- `traits.rs`: `expand` (`keys_mut()[pos]`, inserts at `pos + 1`), `delete_many` (`maxes.remove(prev + 1)`).
- `list.rs`: `append` (`keys_mut()[loc.pos].insert`), `count` (`len() - into_index`, `right.idx - left.idx`, sorted order), `delete` (`len -= 1`, `values[prev]`), `find` (`values.loc`).
- `keylist.rs`: `append` (`values[pos]`, `loc_insert`), `delete` (`len -= 1`, `keys[prev]`, `keys.remove`), `split_values` (`values.insert`).
- `inner.rs`: `Values::loc`, `loc_insert`, `loc_remove`, `loc_len`, `remove_merge` (`remove(pos)`, `remove(1)`, `self[pos - 1]`, `self[0]`), `remove_at`.
- `ops.rs`: `Bisect::new` (`keys()[pos]`), `Expand::new` and `SubListState::new` (`keys()[pos]`), `ForwardSlice::new` (`8 * ...`).
- `bounds.rs`: `try_normalize` (`len().cast_signed()`).
- `search.rs`: `found_in_keys`, `found_in_values`, `increment`.
- `list_update.rs`: `Strategy::from_vec` (`4 * length`).
- `sort/sorted.rs`: `ExtendKind::extend_into` (`Big`: `list.len = length`; `Small`: `keys()[loc.pos]`, `maxes[loc.pos]`, `len += 1`).
- `sort/chunk.rs`: `Batch::extend_into` (`len += idxs.len()`, `chunk_by` runs are never empty, `keys()[last]`, `maxes[last]`), `Chunk::drain_into` (`run[0]`, `len() - run.len()`), `Loc::next_sorted` (`maxes[self.pos + 1..last]`, `keys()[pos][low..]`).
- `sublist.rs`: `SubList::insert_many` (see invariant 15), `remove` and `split_off` (see invariant 2).
- `bisect.rs`: `resolve` (`vec[mid]`).
- `lengths.rs`: `delete_at`, `add_at`, `loc_at` (tree indices on the path to the root), `pos_at` (`pos - offset`), `build` (`inner[size - 1..]` and `split_at_mut(size)` per level).
- `slice.rs`: `fwd_extract` (`start_list[min.idx..max.idx]`), `get_slice` (range bounds), `get_all` (`cast_signed`).
- `iter/builder.rs`: `last_sublist` (`keys()[pos].len()`).
- `iter/impls.rs`: `inner_next` (`+ 1`), `inner_next_back` (`at - 1`).
- `iter/cursor.rs`: `load`, `new_pos`, `From<Loc> for u64` (packing).

### R: runtime (user data, external calls, cache state)

- `traits.rs`: `expand` (`lengths` clear and `expand_at`), `reset` (`Load::try_from`, validated before `clear`).
- `inner.rs`: `set_pos` (`last_len + index`, `index += len`), `set_pos_usize` (`index >= len`, `lengths.build`, `pos_at`).
- `ops.rs`: `Bisect::new` (`is_empty`, `pos == len`), `SubListState::new` (`keys().len() > 1`, and the `Merge` from `pos` and the remaining length), `Locate::new` (`-last < i`).
- `bounds.rs`: `try_normalize` (user inputs), `Loc::into_index` (cache).
- `search.rs`: `on_index` (`MaxesLen`), `index_by_val`, `index_by_key` (bisect and `Placement` results).
- `list_update.rs`: `IntoList::try_new` (`cast_exact`, blocking `read` on the other list, `try_iter`), `IntoList::extend_into` (`list.is_empty()`, `try_collect_vec_with_capacity` consumes the Python iterator), `Strategy::sorted` (`sorted_by_pykey(...)?`).
- `sort/keys.rs`: `KeyFn::call` (the key function can raise).
- `sort/sorted.rs`: `SortedVec::from_iter` (`item?` on every item), `SortedVec::from_vec`.
- `sort/strategy.rs`: `TryCmp::catch` (keeps the first comparison error, the result of the sort is then discarded).
- `sort/chunk.rs`: `LocIdxs::locate` and `Loc::next_sorted` (`rich_compare_bool` and `bisect` call `__lt__`, which can raise).
- `set.rs`: `PySetDataRef::op_mut` (the Python set is mutated before the list is cleared and rebuilt).
- `bisect.rs`: `resolve` (`func(...)?`, `__lt__`).
- `lengths.rs`: `pos_at` (`index - len`, `inner.get(child)` ends the descent).
- `iter/impls.rs`: `inner_next` and `inner_next_back` (`Option` guards).
- `iter/core.rs`: `from_owner` (read lock).
- `core.rs`: `try_read().unwrap()`, `try_write().unwrap()` (lock contention).

### U: unsafe, FFI, external contract

- `bounds.rs`: `From<PySliceIndices>` (`cast_unsigned` relies on the `slice.indices` contract).
- `iter/impls.rs`: `yield_ptr` (`Py_INCREF`).
- `iter/core.rs`: `Deref` (`&*self.list`), `unsafe impl Send` and `Sync`.
- `iter/traits.rs`: `unwrap_unchecked` and `cast_unchecked` (FFI contract).
- `sort/keys.rs`:
  - `SortKey::key` (`Python::assume_attached`: the GIL must be held);
  - `key_cast`, `long`, `float`, `latin` (`cast_unchecked` and unchecked reads, only valid once `KeysCheck` proved the types of every key);
  - `unsafe_object` (calls `tp_richcompare` directly);
- `sort/strategy.rs`: `KeysCheck::check_key` (`cast_unchecked`), `Object::from` (`(*ty).tp_richcompare`).
- `sublist.rs`: `SubList::insert_many` (`ptr::copy`, `ptr::write`, `set_len`), `SubList::last` (`unwrap_unchecked`, relies on invariant 2).
