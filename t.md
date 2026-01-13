# Benchmark results between new and old Option/Result implementations

Basic operations like `is_some`, `unwrap`, `xor`, etc., are virtually identical in performance between the new Rust-backed implementation and the old pure-Python one, and are not shown here.

This is expected, as these operations were already just returnining boolean values or attributes, which can't be significantly optimized further in a an interpreted language like Python (no inlining).

However, more complex operations that involve higher-order functions and chaining show significant performance improvements.
The `eq` method for example, which is just this in pure python:

```python
    def eq(self, other: Option[T]) -> bool:
        if other.is_some():
            return self.value == other.unwrap()
        return False
```

Saw a 74% speedup when implemented in Rust.

This is great because the Option/Results operations were already cheap in themselve (a few boolean checks and methods calls at most), so any speedup here directly translates to better overall performance in any code using them, and open the door for more improvements and ports to Rust in the future.

## Observed Improvements

1. A flat 2x speedup on instanciation for direct creation `Some(value)`, and improved dispatch times when calling methods on the Option constructor itself. Improvement for ALL code using Options/Results.
2. Equality checks see a nice boost on the dunder method, who was doing runtime checks to correctly dispatch to the right implementation based on the type of the other operand.
This means that checking equality is now *cheap*, and using the plain `eq` method also got a nice boost as mentionned earlier.
3. Map operations got a speedup of +100%, thanks to carefully optimized parameter passing and closure handling in Rust.
4. Speedups of over **4x** and up to **10x** were observed in operations like `flatten`, `unzip`, or `transpose`, who were doing a few more steps. Even in pure Python, this was cheap, but any operation done in Rust is now significantly faster, which compounds the more work is done.
5. Iterator working on Option/Result types benefit from this as well, as this can be seen in the `Iter.map(Option)` (4.41x speedup!),  and the very common `Iter.filter_map` (2.11x speedup) operations.
Note that filter_map logic will be ported to Rust as well, which will bring even more improvements in the future. fast paths on maps could maybe be implemented as well, this remains to be seen.

```shell
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Category           ┃ Operation                     ┃ Rust (s, median) ┃ Python (s, median) ┃ Speedup ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ Instantiation      │ Some(value)                   │           0.0001 │             0.0002 │   2.27x │
│ Instantiation      │ Dispatch to Some              │           0.0001 │             0.0004 │   3.28x │
│ Instantiation      │ Dispatch to None              │           0.0001 │             0.0001 │   1.53x │
│ Equality Checks    │ __eq__                        │           0.0001 │             0.0002 │   3.35x │
│ Equality Checks    │ eq_method                     │           0.0001 │             0.0001 │   1.74x │
│ Map with Closures  │ map (identity)                │           0.0002 │             0.0004 │   2.44x │
│ Map with Closures  │ map simple add                │           0.0002 │             0.0004 │   2.39x │
│ Chained Operations │ map -> filter -> map          │           0.0001 │             0.0002 │   1.72x │
│ Iter with Options  │ Iter.map(Option)              │           0.0005 │             0.0024 │   4.41x │
│ Iter with Options  │ Iter.filter_map (simple)      │           0.0044 │             0.0093 │   2.11x │
│ Iter with Options  │ Iter.map -> filter_map -> map │           0.0057 │             0.0183 │   3.22x │
│ Complex Methods    │ flatten                       │           0.0001 │             0.0003 │   4.93x │
│ Complex Methods    │ unzip                         │           0.0001 │             0.0005 │   4.32x │
│ Complex Methods    │ zip                           │           0.0001 │             0.0003 │   3.25x │
│ Complex Methods    │ zip_with                      │           0.0001 │             0.0004 │   2.49x │
│ Complex Methods    │ transpose (Option->Result)    │           0.0001 │             0.0010 │  10.21x │
│ Complex Methods    │ transpose (Result->Option)    │           0.0001 │             0.0009 │   8.98x │
└────────────────────┴───────────────────────────────┴──────────────────┴────────────────────┴─────────┘

Median speedup: 3.22x
Rust wins: 17/17
```
