from collections.abc import Mapping
from pprint import pformat
from typing import Any


def dict_repr(
    v: Mapping[Any, Any],
    max_items: int = 20,
    depth: int = 3,
    width: int = 80,
    *,
    compact: bool = True,
) -> str:
    truncated = dict(list(v.items())[:max_items])
    suffix = "..." if len(v) > max_items else ""
    return pformat(truncated, depth=depth, width=width, compact=compact) + suffix
