# Benchmark results

This document presents the benchmark results comparing the current implementation of option and result pipelines with a new implementation referred to as `new_pc`. The benchmarks were conducted using various test cases to evaluate performance improvements.

[Code for running the benchmarks](`scripts/bench_option_current_vs_third.py`)

To execute them, use the following command at the root of the repository:

```shell
uv run -m scripts.bench_option_current_vs_third
```

**Results**:

```shell
PS C:\Users\tibo\python_codes\pyochain> uv run -m scripts.bench_option_current_vs_third
              Option & Result pipelines: current vs new_pc (median + Q1-Q3 range)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Case                         ┃ Current (ms) ┃ New PC (ms) ┃ Improvement % ┃      Q1-Q3 Range ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ option/chained_zips          │   163.459 ms │  138.128 ms │       -13.57% │  -21.8% to -4.8% │
│ option/complex               │    84.471 ms │   65.548 ms │       -30.77% │ -47.6% to -27.7% │
│ option/deep_nesting          │    99.459 ms │   90.288 ms │       -14.07% │  -20.6% to -9.4% │
│ option/long_chain            │   201.514 ms │  176.545 ms │       -11.96% │ -15.5% to -10.1% │
│ option/mixed_ops             │   155.094 ms │  136.473 ms │       -13.58% │  -31.2% to -4.0% │
│ option/multi_validation      │    85.255 ms │   72.250 ms │       -16.76% │  -31.3% to -2.8% │
│ result/deep_chain            │   213.313 ms │  185.063 ms │       -12.52% │  -14.6% to -7.0% │
│ result/is_ok_and_filtering   │   154.526 ms │  147.439 ms │        -3.59% │   -10.7% to 1.2% │
│ result/mixed                 │   255.877 ms │  244.462 ms │        -7.34% │  -13.2% to -2.3% │
│ result/with_option_transpose │   120.270 ms │  110.766 ms │        -9.00% │  -23.5% to -0.0% │
└──────────────────────────────┴──────────────┴─────────────┴───────────────┴──────────────────┘
PS C:\Users\tibo\python_codes\pyochain> uv run -m scripts.bench_option_current_vs_third
              Option & Result pipelines: current vs new_pc (median + Q1-Q3 range)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Case                         ┃ Current (ms) ┃ New PC (ms) ┃ Improvement % ┃      Q1-Q3 Range ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ option/chained_zips          │   165.672 ms │  141.419 ms │       -19.25% │ -25.8% to -11.6% │
│ option/complex               │    80.643 ms │   73.001 ms │       -11.60% │  -16.7% to -9.4% │
│ option/deep_nesting          │   106.043 ms │   89.391 ms │       -20.18% │ -28.7% to -18.3% │
│ option/long_chain            │   211.906 ms │  178.759 ms │       -16.56% │ -21.4% to -14.8% │
│ option/mixed_ops             │   159.363 ms │  127.600 ms │       -24.95% │ -31.4% to -17.2% │
│ option/multi_validation      │    85.111 ms │   68.387 ms │       -24.55% │  -33.2% to -8.0% │
│ result/deep_chain            │   210.313 ms │  186.520 ms │        -8.22% │  -12.2% to -4.4% │
│ result/is_ok_and_filtering   │   161.344 ms │  148.017 ms │        -5.83% │  -12.1% to -0.0% │
│ result/mixed                 │   270.255 ms │  238.954 ms │       -12.96% │ -17.4% to -11.1% │
│ result/with_option_transpose │   122.572 ms │  110.764 ms │       -12.50% │  -17.8% to -6.1% │
└──────────────────────────────┴──────────────┴─────────────┴───────────────┴──────────────────┘
PS C:\Users\tibo\python_codes\pyochain>
```

## Conclusion

In these benchmarks, the new implementation consistently outperforms the current implementation across various test cases. The performance improvements range from approximately 3.59% to 30.77%, indicating that the new approach is generally more efficient. This suggests that adopting the new implementation could lead to significant performance gains in applications utilizing these option and result pipelines.
