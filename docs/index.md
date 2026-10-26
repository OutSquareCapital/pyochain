# pyochain ⛓️

**_Functional-style method chaining for Python data structures._**

Welcome to the `pyochain` documentation! This library brings a fluent, declarative API inspired by Rust and DataFrame libraries to your Python iterables and dictionaries.

## Installation

```bash
uv add pyochain
```

## Quick start

```python
from pyochain import Iter, Seq

res = (
    Iter
    .from_count(1)
    .filter(lambda x: x % 2 != 0)
    .map(lambda x: x**2)
    .take(5)
    .collect(Seq)
)
assert res == (1, 9, 25, 49, 81)
```

## Next steps

- [**Overview**](overview.md)
- [**API Reference**](api-reference.md) — complete public API docs
- [🤖 Agent-friendly API reference](https://outsquarecapital.github.io/pyochain/llms.txt)
- [🤖 Agent-friendly API reference (All content at once)](https://outsquarecapital.github.io/pyochain/llms.txt)
- [**GitHub Repository**](https://github.com/OutSquareCapital/pyochain)
- [**Contributing Guide**](https://github.com/OutSquareCapital/pyochain/blob/master/CONTRIBUTING.md)
- [**PyPI Package**](https://pypi.org/project/pyochain/)
