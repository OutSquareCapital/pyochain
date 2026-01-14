# Temp changelog

1. Deleted `Iter.join_with` method
2. Performance improvement on {Option, Result}.expect_* methods in Rust bindings -> func was signed with `String` before, uncessessarily converting python str to rust String. Benchmarks show performance boost of +126% on these methods
3. Better use of Pyo3 lifetimes with bounds, resulting in simpler code and a flat 1% performance gain across the board
4. Added Checkable and Pipeable in Rust bindings, made them Protocols in stubs

Benchmark details for `Result.ok_or_else` (False case) and `Result.then` (True case) improvements:

```shell
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Category   ┃ Operation    ┃ Rust (s, median) ┃ Python (s, median) ┃ Speedup ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ ok_or_else │ no_args      │           0.0004 │             0.0005 │   1.27x │
│ ok_or_else │ args         │           0.0005 │             0.0006 │   1.24x │
│ ok_or_else │ kwargs       │           0.0006 │             0.0007 │   1.13x │
│ then       │ no_args_then │           0.0004 │             0.0005 │   1.26x │
│ then       │ args_then    │           0.0005 │             0.0006 │   1.24x │
│ then       │ kwargs_then  │           0.0006 │             0.0007 │   1.11x │
└────────────┴──────────────┴──────────────────┴────────────────────┴─────────┘

Median speedup: 1.24x
Rust wins: 6/6
```

We test those case as the other one don't do much (just wrap in `Ok` or `Some` where the initializer is already in Rust).
We can see that the speedup is consistent across the board, with a speed-up of around +25% with no args or positional args, and around +10% with keyword arguments.
For reference, python implementation of `ok_or_else`:

```python
    def ok_or_else[**P, E](
        self,
        func: Callable[Concatenate[Self, P], E],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[Self, E]:
    return Ok(self) if self else Err(func(self, *args, **kwargs))
```

"True" work only happen on `Err` case.
