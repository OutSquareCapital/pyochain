from collections.abc import Iterable, Iterator, Mapping
from pprint import pformat
from typing import Any, NamedTuple

import cytoolz as cz


class Peeked[T](NamedTuple):
    value: T | tuple[T, ...]
    sequence: Iterator[T]


def peekn[T](data: Iterable[T], n: int) -> Iterator[T]:
    peeked = Peeked(*cz.itertoolz.peekn(n, data))
    print(f"Peeked {n} values: {peeked.value}")
    return peeked.sequence


def peek[T](data: Iterable[T]) -> Iterator[T]:
    peeked = Peeked(*cz.itertoolz.peek(data))
    print(f"Peeked value: {peeked.value}")
    return peeked.sequence


def dict_repr(
    v: Mapping[Any, Any],
    max_items: int = 20,
    depth: int = 3,
    width: int = 80,
    compact: bool = True,
) -> str:
    truncated = dict(list(v.items())[:max_items])
    suffix = "..." if len(v) > max_items else ""
    return pformat(truncated, depth=depth, width=width, compact=compact) + suffix
