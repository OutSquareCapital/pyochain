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
