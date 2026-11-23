# sorted_rs

This crate aims to centralize as much as possible the logic related to sorted containers.

The main crate then handle the bridge between the "pure rust" logic and the Python interface.

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

Notation: `values` holds the elements, `keys` holds the sort keys (for `List`, `keys` and `values` are the same storage), `maxes` holds the last key of each chunk, `load` is the load factor.

### Structural

#### 1. Lockstep

- Invariant: `maxes.len() == keys.len() == values.len()`.
- Used by: `Bisect::new`, `expand`, `push_val`, `index_to_loc`, `islice_bounds`.
- Checked in `debug.rs`: yes.

#### 2. Non-empty chunks

- Invariant: every chunk in `keys` and `values` has at least one element.
- Used by: `.last().unwrap()` in `traits.rs` (`expand`, `delete_on_len_sup_to_half_load`), `sort/chunk.rs` (`Chunk::push_into`, `Batch::extend_into`), `list.rs` (`delete`), `keylist.rs` (`delete`), `inner.rs` (`loc_last`, `index_to_loc`), `ops.rs` (`Locate`).
- Checked in `debug.rs`: no. The checker only asserts `len >= lower_limit` for every chunk except the last. An empty last chunk is only caught indirectly, by a panic of the `.last().unwrap()` in the checker itself.

#### 3. Max mirror

- Invariant: `maxes[i] == keys[i].last()` for every `i`.
- Maintained by hand in: `expand`, `delete`, `push_val`, `delete_on_len_sup_to_half_load`, `Chunk::push_into`, `Batch::extend_into` and the `Small` arm of `ExtendKind::extend_into`.
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

- Invariant: `len <= 2 * load` for every chunk, and `len >= load / 2` for every chunk except the last.
- Enforced by: the upper bound at insert time (`Expand::new`, and the `while` loop of `Batch::extend_into`, which splits as many times as needed), the lower bound at delete time (`SubListState::new`). `Big` builds chunks of exactly `load` items, except the last.
- Checked in `debug.rs`: yes.

#### 7. Length tree (`lengths`)

- Invariant: either empty (not built), or consistent with `values`: `inner[0] == len`, `inner.len() == offset + values.len()`, and each internal node equals the sum of its children.
- Built lazily on the first index access past the first chunk (`set_pos_usize`, `into_index`). For a single chunk, `[len]` with `offset = 0`, as in the original.
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

#### 14. Load factor

- `load >= 4` (`Load::test_limit`) comes from the original `sortedcontainers` test suite. `debug.rs` asserts it, but it has not been established that the implementation requires it.
- `reset` does not validate it. `load == 0` panics on the first split, and `ExtendKind::Big` panics on `length.div_ceil(load)` after having set `len`, leaving `len` and `values` desynchronized.
- `2 * load` and the cursor packing (`u32` for `pos` and `idx`) must not overflow.

### Sorted batches

#### 15. Sorted input

- Invariant: the items of a `SortedVec` are in non-decreasing order of their key, and `Loc::next_sorted` is only called on a non-empty list.
- Established by: `SortedVec::from_iter` and `SortedVec::from_vec` (sort), and `IntoList::extend_into`, which only picks `Medium` and `Small` when the list isn't empty.
- Relied on by: `Loc::next_sorted` (compares from the previous location, never goes back), `Batch::locate`, the `Small` arm of `ExtendKind::extend_into`.

#### 16. Batch application

- Invariant: every location of a `Batch` is computed against the list before any insertion, then applied from the last sublist to the first, so a split never shifts a location that wasn't applied yet.
- `Vec::insert_many` (`unsafe`: `ptr::copy`, `ptr::write`, `set_len`) requires one location per item, with sorted indices, each `<= len`. Asserted, and covered by unit tests in `sort/chunk.rs`.

### Known gaps

- `core.rs` uses `try_read().unwrap()` / `try_write().unwrap()`, which panics on lock contention, and on any re-entrant call made while the lock is held.
- `IntoList::PyIter` is consumed lazily, inside `ListsMethods::extend`. The Python code of the iterable, of the key function and of `__lt__` therefore runs while the write lock is held.
- Some updates aren't atomic:
  - the `Small` arm of `ExtendKind::extend_into` compares and inserts item by item, so an exception leaves a partial update;
  - `imul`, `reset`, `delete_fwd_small`, `Dict::update` and `PySetDataRef::op_mut` call `clear` and then re-sort, so an exception in the second step leaves the list empty (for `op_mut`, with the Python set already updated).
  - `Medium` locates every item before mutating anything, and `Big` sorts before `clear`: both are atomic.
- `iter/core.rs` marks `InnerRef` as `Send` and `Sync`, and releases the read lock before iterating. Mutating the list while an iterator is alive is a data race in Rust terms; only the GIL currently prevents it.

## Statically known runtime checks

Note: Wrote by Claude clanker, not rechecked toroughly. Planned to be used as a TODO list.

Runtime checks whose outcome is already fixed by the type or by the call site. Each group below is a candidate for encoding, so the check can disappear. The sites are listed by file and function, with the line number at the time of writing.

### Witness `NonEmpty` chunk

A chunk in `keys` or `values` that is never empty. Encoding it removes the `last().unwrap()` calls.

| Location                                                               | Code                                   | Note                            |
| ---------------------------------------------------------------------- | -------------------------------------- | ------------------------------- |
| `traits.rs`, l.[63](src/traits.rs#L63)                                 | `half.last().unwrap()`                 | in `expand`                     |
| `traits.rs`, l.[65](src/traits.rs#L65)                                 | `self.keys()[pos].last().unwrap()`     | in `expand`                     |
| `traits.rs`, l.[96](src/traits.rs#L96)                                 | `self.keys().loc_last(&loc)`           | [^1]                            |
| `sort/chunk.rs`, l.[36](src/sort/chunk.rs#L36)                         | `self.last().unwrap()`                 | in `Chunk::push_into`           |
| `sort/chunk.rs`, l.[154](src/sort/chunk.rs#L154)                       | `list.keys()[last].last().unwrap()`    | in `Batch::extend_into`         |
| `list.rs`, l.[95](src/list.rs#L95)                                     | `self.values[prev].last().unwrap()`    | in `delete`                     |
| `list.rs`, l.[99](src/list.rs#L99)                                     | `self.keys().loc_last(&loc)`           | in `delete`                     |
| `keylist.rs`, l.[97](src/keylist.rs#L97)                               | `left[prev].last().unwrap()`           | in `delete`                     |
| `keylist.rs`, l.[102](src/keylist.rs#L102)                             | `self.keys().loc_last(&loc)`           | in `delete`                     |
| `inner.rs`, l.[85](src/inner.rs#L85)                                   | `self[loc.pos].last().unwrap()`        | in `Values::loc_last`           |
| `inner.rs`, l.[134](src/inner.rs#L134)                                 | `self.values[pos].len() - 1_usize`     | in `index_to_loc`, `Last`       |
| `inner.rs`, l.[141](src/inner.rs#L141)                                 | `self.values[pos].len()`               | in `index_to_loc`, `InLastList` |
| `inner.rs`, l.[166](src/inner.rs#L166)                                 | `self.values[pos].len()`               | in `set_pos`                    |
| `ops.rs`, l.[178](src/ops.rs#L178)                                     | `list.values.last().unwrap()`          | in `Locate::new`                |
| `search.rs`, l.[55](src/search.rs#L55)                                 | `ops::Bisect::Empty => unreachable!()` | [^2]                            |
| `lengths.rs`, l.[99](src/lengths.rs#L99), l.[101](src/lengths.rs#L101) | `tree.last().unwrap()`                 | [^3]                            |
| `iter/impls.rs`, l.[80](src/iter/impls.rs#L80)                         | `(v.len() - 1) as u64`                 | [^4]                            |
| `iter/builder.rs`, l.[92](src/iter/builder.rs#L92)                     | `values[pos].len()`                    | in `islice_bounds`              |

[^1]: in `delete_on_len_sup_to_half_load`, guarded by `LenSupToHalfLoad`

[^2]: in `Search::on_index`; the chunk is non-empty after `try_normalize`

[^3]: in `build`; never `None`, since `tree` starts with two rows

[^4]: in `inner_next_back`; `v` comes from `last()`

### Witness `NonEmpty` list

A list with at least one chunk (`maxes` and `values` non-empty). Encoding it removes the `unreachable!()` calls and the `len() - 1` on `maxes` and `values`.

| Location                                             | Code                                   | Note                            |
| ---------------------------------------------------- | -------------------------------------- | ------------------------------- |
| `list.rs`, l.[71](src/list.rs#L71)                   | `ops::Bisect::Empty => unreachable!()` | [^6]                            |
| `search.rs`, l.[55](src/search.rs#L55)               | `ops::Bisect::Empty => unreachable!()` | [^7]                            |
| `traits.rs`, l.[136](src/traits.rs#L136)             | `self.maxes.len() - 1`                 | [^8]                            |
| `keylist.rs`, l.[53](src/keylist.rs#L53)             | `self.maxes.len() - 1`                 | [^9]                            |
| `sort/chunk.rs`, l.[185](src/sort/chunk.rs#L185)     | `list.maxes.len() - 1`                 | [^25]                           |
| `inner.rs`, l.[133](src/inner.rs#L133)               | `self.values.len() - 1`                | in `index_to_loc`, `Last`       |
| `inner.rs`, l.[140](src/inner.rs#L140)               | `self.values.len() - 1`                | in `index_to_loc`, `InLastList` |
| `inner.rs`, l.[165](src/inner.rs#L165)               | `self.values.len() - 1`                | in `set_pos`                    |
| `inner.rs`, l.[181](src/inner.rs#L181)               | `self.values[0].len()`                 | in `set_pos_usize`              |
| `ops.rs`, l.[175](src/ops.rs#L175)                   | `list.values[0].len()`                 | [^10]                           |
| `ops.rs`, l.[177](src/ops.rs#L177)                   | `list.values[0].len()`                 | [^11]                           |
| `slice.rs`, l.[117](src/slice.rs#L117)               | `list.values.len() - 1`                | in `fwd_extract`                |
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
| `ops.rs`, l.[195](src/ops.rs#L195) to l.[200](src/ops.rs#L200)      | `index < bounds.start`                                 | [^13]                         |
| `slice.rs`, l.[47](src/slice.rs#L47)                                | `ops::ForwardSlice::new(self.indices, self.ref_len)`   | in `extract_from`, `Backward` |
| `slice.rs`, l.[107](src/slice.rs#L107)                              | `list.set_pos_usize(slf.start)?`                       | in `fwd_extract`              |
| `slice.rs`, l.[122](src/slice.rs#L122)                              | `list.set_pos_usize(slf.stop)?`                        | in `fwd_extract`              |
| `slice.rs`, l.[108](src/slice.rs#L108)                              | `Loc::with_idx(min.idx + (slf.stop - slf.start))`      | in `fwd_extract`              |
| `slice.rs`, l.[163](src/slice.rs#L163)                              | `list.set_pos_usize(idx)?`                             | in `del_all`                  |
| `slice.rs`, l.[145](src/slice.rs#L145)                              | `reset_slice((0, self.start), self.ref_len)`           | [^14]                         |
| `slice.rs`, l.[148](src/slice.rs#L148)                              | `reset_slice((self.stop, self.ref_len), self.ref_len)` | [^15]                         |
| `bounds.rs`, l.[19](src/bounds.rs#L19) to l.[23](src/bounds.rs#L23) | `if i.step > 0 { .. } else { .. }`                     | [^27]                         |
| `bounds.rs`, l.[54](src/bounds.rs#L54)                              | `Bounds::new(start, stop.cast_unsigned())`             | [^16]                         |

[^13]: in `Placement::new`; ordering comparisons

[^14]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^15]: in `delete_fwd_small`; empty cases handled by `NoOp`

[^16]: in `try_normalize`; this is where the witness is created for `index` and `islice`

[^27]: in `From<PySliceIndices> for Bounds`; this is where the witness is created for slices, with the `start >= stop => NoOp` guard of `Slice::new` covering the empty case

### Witness `Load` (`load >= 1`, bounded)

A load factor validated at construction: `>= 1`, with `2 * load` and `load / 2` precomputed without overflow. Encoding it removes the implicit assumptions in the chunk size checks.

| Location                                         | Code                                                | Note                   |
| ------------------------------------------------ | --------------------------------------------------- | ---------------------- |
| `inner.rs`, l.[29](src/inner.rs#L29)             | `self.0 * 2`                                        | [^17]                  |
| `inner.rs`, l.[33](src/inner.rs#L33)             | `self.0 / 2`                                        | in `Load::lower_limit` |
| `ops.rs`, l.[124](src/ops.rs#L124)               | `data.load.higher_limit()`                          | in `Expand::new`       |
| `ops.rs`, l.[145](src/ops.rs#L145)               | `data.load.lower_limit()`                           | in `SubListState::new` |
| `traits.rs`, l.[63](src/traits.rs#L63)           | `half.last().unwrap()`                              | [^18]                  |
| `sort/sorted.rs`, l.[61](src/sort/sorted.rs#L61) | `let load = *list.load;`                            | [^19]                  |
| `sort/sorted.rs`, l.[70](src/sort/sorted.rs#L70) | `.take(length.div_ceil(load))`                      | [^28]                  |
| `iter/cursor.rs`, l.[91](src/iter/cursor.rs#L91) | `(loc.pos as Self) << u32::BITS \| loc.idx as Self` | [^20]                  |

[^17]: in `Load::higher_limit`; `2 * load` overflow

[^18]: in `expand`; non-empty only if `load >= 1`

[^19]: in `ExtendKind::extend_into`, `Big`; `take(load)` yields non-empty chunks only if `load >= 1`

[^28]: in `ExtendKind::extend_into`, `Big`; `div_ceil` panics if `load == 0`, after `len` was already set

[^20]: in `From<Loc> for u64`; `idx < 2^32` only if chunk length is bounded by `2 * load`

### Witness `NonZeroIsize` (step)

| Location                             | Code                                             | Note  |
| ------------------------------------ | ------------------------------------------------ | ----- |
| `ops.rs`, l.[53](src/ops.rs#L53)     | `Nb::Zero => unsafe { unreachable_unchecked() }` | [^21] |
| `slice.rs`, l.[26](src/slice.rs#L26) | `let step = i.step.into();`                      | [^22] |

[^21]: in `Slice::new`; removing the variant removes the `unsafe`

[^22]: in `SliceIdxs::new`; the place where the witness is created

### Witness `Split` (`MustSplit` guard)

| Location                                   | Code                                   | Note  |
| ------------------------------------------ | -------------------------------------- | ----- |
| `traits.rs`, l.[62](src/traits.rs#L62)     | `self.keys_mut()[pos].split_off(load)` | [^23] |
| `keylist.rs`, l.[124](src/keylist.rs#L124) | `self.values[pos].split_off(load)`     | [^24] |

[^23]: in `expand`; guarded by `keys()[pos].len() > higher_limit()`

[^24]: in `split_values`; same guard, shared with `keys`

### Delete path witness

| Location                                 | Code                      | Note  |
| ---------------------------------------- | ------------------------- | ----- |
| `traits.rs`, l.[105](src/traits.rs#L105) | `let prev = loc.pos - 1;` | [^26] |

[^26]: in `delete_many`; guarded by `ManySublists`, which implies at least two chunks

## Other classified sites

Sites that are not statically encodable as above. They are kept here so the inventory is complete.

### I: internal invariant

Depends on the structure (lockstep, Loc validity, sorted order), not on a type that could hold it.

- `traits.rs`: `expand` (`keys_mut()[pos]`, inserts at `pos + 1`), `delete_many` (`values[prev]`).
- `list.rs`: `append` (`keys_mut()[loc.pos].insert`), `count` (`len() - into_index`, `right.idx - left.idx`, sorted order), `delete` (`len -= 1`), `find` (`values.loc`).
- `keylist.rs`: `append` (`values[pos]`, `loc_insert`), `delete` (`len -= 1`, `split_at_mut`, `left[prev]`, `right[0]`, `keys.remove`), `split_values` (`values.insert`).
- `inner.rs`: `Values::loc`, `loc_insert`, `loc_remove`, `loc_len`, `remove_at`.
- `ops.rs`: `Bisect::new` (`keys()[pos]`), `Expand::new` and `SubListState::new` (`keys()[pos]`), `ForwardSlice::new` (`8 * ...`).
- `bounds.rs`: `try_normalize` (`len().cast_signed()`).
- `search.rs`: `found_in_keys`, `found_in_values`, `increment`.
- `list_update.rs`: `Strategy::from_vec` (`4 * length`).
- `sort/sorted.rs`: `ExtendKind::extend_into` (`Big`: `list.len = length`; `Small`: `keys()[loc.pos]`, `maxes[loc.pos]`, `len += 1`).
- `sort/chunk.rs`: `Batch::extend_into` (`len += idxs.len()`, `chunk_by` runs are never empty, `keys()[last]`, `maxes[last]`), `Chunk::drain_into` (`run[0]`, `len() - run.len()`), `Loc::next_sorted` (`maxes[self.pos + 1..last]`, `keys()[pos][low..]`), `Vec::insert_many` (see invariant 16).
- `bisect.rs`: `resolve` (`vec[mid]`).
- `lengths.rs`: `delete_at`, `expand_at`, `loc_at` (tree indices), `pos_at` (`pos - offset`).
- `slice.rs`: `fwd_extract` (`start_list[min.idx..max.idx]`), `get_slice` (range bounds), `get_all` (`cast_signed`).
- `iter/builder.rs`: `last_sublist` (`keys()[pos].len()`).
- `iter/impls.rs`: `inner_next` (`+ 1`), `inner_next_back` (`at - 1`).
- `iter/cursor.rs`: `load`, `new_pos`, `From<Loc> for u64` (packing).

### R: runtime (user data, external calls, cache state)

- `traits.rs`: `expand` (`lengths` clear and `expand_at`), `reset` (`load.into()`).
- `inner.rs`: `set_pos` (`last_len + index`, `index += len`), `set_pos_usize` (`index >= len`, `lengths.build`, `pos_at`).
- `ops.rs`: `Bisect::new` (`is_empty`, `pos == len`), `SubListState::new` (`keys().len() > 1`), `Locate::new` (`-last < i`).
- `bounds.rs`: `try_normalize` (user inputs), `Loc::into_index` (cache).
- `search.rs`: `on_index` (`MaxesLen`), `index_by_val`, `index_by_key` (bisect and `Placement` results).
- `list_update.rs`: `IntoList::try_new` (`cast_exact`, blocking `read` on the other list, `try_iter`), `IntoList::extend_into` (`list.is_empty()`, `try_collect_vec_with_capacity` consumes the Python iterator), `Strategy::sorted` (`sorted_by_pykey(...)?`).
- `sort/keys.rs`: `KeyFn::call` (the key function can raise).
- `sort/sorted.rs`: `SortedVec::from_iter` (`item?` on every item), `SortedVec::from_vec`.
- `sort/strategy.rs`: `TryCmp::catch` (keeps the first comparison error, the result of the sort is then discarded).
- `sort/chunk.rs`: `LocIdxs::locate` and `Loc::next_sorted` (`rich_compare_bool` and `bisect` call `__lt__`, which can raise).
- `set.rs`: `PySetDataRef::op_mut` (the Python set is mutated before the list is cleared and rebuilt).
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
- `sort/keys.rs`:
  - `SortKey::key` (`Python::assume_attached`: the GIL must be held);
  - `key_cast`, `long`, `float`, `latin` (`cast_unchecked` and unchecked reads, only valid once `KeysCheck` proved the types of every key);
  - `unsafe_object` (calls `tp_richcompare` directly);
  - `tup_cmp` (`get_borrowed_item_unchecked`, bounded by the `len` checks of the loop).
- `sort/strategy.rs`: `KeysCheck::check_key` (`cast_unchecked`), `Object::from` (`(*ty).tp_richcompare`).
- `sort/chunk.rs`: `Vec::insert_many` (`ptr::copy`, `ptr::write`, `set_len`).
