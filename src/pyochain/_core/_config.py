import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self


@dataclass(slots=True)
class PyochainConfig:
    max_items: int = 20

    def set_max_items(self, max_items: int) -> Self:
        self.max_items = max_items
        return self

    def iter_repr(self, v: Iterable[Any]) -> str:
        from pprint import pformat

        if isinstance(v, Iterator):
            return v.__repr__()
        items = (
            v if isinstance(v, Sequence) else tuple(itertools.islice(v, self.max_items))
        )

        # Empty case
        if not items:
            return ""
        return _strip_inner_container(pformat(items, sort_dicts=False))

    def dict_repr(self, v: Mapping[Any, Any]) -> str:
        from pprint import pformat

        return pformat(v, sort_dicts=False)


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
