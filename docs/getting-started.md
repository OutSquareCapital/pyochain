# Getting Started

## Installation

```bash
uv add pyochain
```

## Quick start

```python
>>> import pyochain as pc
>>> res: pc.Seq[int] = (
...     pc.Iter.from_count(1)
...     .filter(lambda x: x % 2 != 0)
...     .map(lambda x: x**2)
...     .take(5)
...     .collect()
... )
>>> res
Seq(1, 9, 25, 49, 81)
```

## Next steps

- [User Guide](user-guide.md): concepts and mental model
- [Core Types Overview](core-types-overview.md): choose between `Iter`, `Seq`, `Vec`, `Set`, `Dict`, `Option`, `Result`
- [Examples & Cookbook](examples.md): practical patterns
