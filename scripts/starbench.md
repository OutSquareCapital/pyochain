# Result of benmchmarks for map_star vs map with lambda unpacking

```shell
================================================================================
TEST 1: 2-argument function (make_sku)
================================================================================
  Tuples (lambda x: func(*x))         : 0.0783s
  NamedTuples (lambda x: func(*x))    : 0.0991s
  map_star(func)                      : 0.0663s
  📊 Tuple vs map_star                : 15.39% faster
  📊 NamedTuple vs map_star           : 33.13% faster

================================================================================
TEST 2: 3-argument function (process_triple)
================================================================================
  Tuples (lambda x: func(*x))         : 0.2035s
  NamedTuples (lambda x: func(*x))    : 0.3252s
  map_star(func)                      : 0.1320s
  📊 Tuple vs map_star                : 35.13% faster
  📊 NamedTuple vs map_star           : 59.39% faster

================================================================================
TEST 3: Mixed types (format_record)
================================================================================
  Tuples (lambda x: func(*x))         : 0.6436s
  NamedTuples (lambda x: func(*x))    : 0.6865s
  map_star(func)                      : 0.5064s
  📊 Tuple vs map_star                : 21.31% faster
  📊 NamedTuple vs map_star           : 26.24% faster

================================================================================
TEST 4: Heavy computation (5 arguments)
================================================================================
  Tuples (lambda x: func(*x))         : 0.2572s
  NamedTuples (lambda x: func(*x))    : 0.3423s
  map_star(func)                      : 0.1705s
  📊 Tuple vs map_star                : 33.71% faster
  📊 NamedTuple vs map_star           : 50.18% faster

================================================================================
SUMMARY
================================================================================

✅ Average improvement with map_star:
   vs Tuples + lambda: 28.22%
   vs NamedTuples + lambda: 41.78%
```
