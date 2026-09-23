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
