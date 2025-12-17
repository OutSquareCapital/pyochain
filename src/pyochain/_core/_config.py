import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self


@dataclass(slots=True)
class PyochainConfig:
    max_items: int = 20
    depth: int = 3
    width: int = 80
    compact: bool = True
    """Enable rich formatting if available."""

    def set_width(self, width: int) -> Self:
        self.width = width
        return self

    def set_depth(self, depth: int) -> Self:
        self.depth = depth
        return self

    def set_max_items(self, max_items: int) -> Self:
        self.max_items = max_items
        return self

    def set_compact(self, *, compact: bool) -> Self:
        self.compact = compact
        return self

    def _truncate[T](self, v: Iterable[T]) -> tuple[T, ...]:
        return tuple(itertools.islice(v, self.max_items))

    def iter_repr(self, v: Iterable[Any]) -> str:
        from pprint import pformat

        # Convert to sequence if needed for inspection
        if isinstance(v, Iterator):
            return v.__repr__()
        if isinstance(v, Sequence):
            items = v
            is_truncated = len(v) > self.max_items
        else:
            items = self._truncate(v)
            is_truncated = False

        # Empty case
        if not items:
            return ""
        formatted = pformat(
            items[: self.max_items] if is_truncated else items,
            depth=self.depth,
            width=self.width,
            compact=self.compact,
            sort_dicts=False,
        )

        suffix = ", ..." if is_truncated else ""
        return _strip_inner_container(formatted) + suffix

    def dict_repr(self, v: Mapping[Any, Any]) -> str:
        from pprint import pformat

        if not v:
            return "{}"

        is_truncated = len(v) > self.max_items
        suffix = "..." if is_truncated else ""
        return (
            pformat(
                dict(list(v.items())[: self.max_items]),
                depth=self.depth,
                width=self.width,
                compact=self.compact,
            )
            + suffix
        )


def _strip_inner_container(formatted: str) -> str:
    """Strip outer container characters.

    pformat returns things like: (1, 2, 3) or [1, 2, 3] or (1,), We want to remove the outer delimiters.

    Avoid repetition with pyochain own delimiters.
    """
    if (formatted.startswith("(") and formatted.endswith(")")) or (
        formatted.startswith("[") and formatted.endswith("]")
    ):
        return formatted[1:-1]
    return formatted


_CONFIG = PyochainConfig()
"""Global Pyochain configuration.
Allow to customize the representation of various Pyochain types.
"""


def get_config() -> PyochainConfig:
    """Get the global Pyochain configuration."""
    return _CONFIG
