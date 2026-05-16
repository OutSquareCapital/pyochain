# Getting Started

## Installation

```bash
uv add pyochain
```

## Quick start

```python
>>> from pyochain import Iter, Seq
>>> res: Seq[int] = (
...     Iter.from_count(1)
...     .filter(lambda x: x % 2 != 0)
...     .map(lambda x: x**2)
...     .take(5)
...     .collect()
... )
>>> res
Seq(1, 9, 25, 49, 81)
```

## Next steps

- [Core Types Overview](core-types-overview.md): choose between the various provided types
- [Interoperability](interoperability.md): convert between types with various methods
