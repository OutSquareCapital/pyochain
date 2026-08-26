from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, assert_type

from pyochain import (
    Iter,
    Range,
)
from pyochain.abc import PyoIterator

if TYPE_CHECKING:
    from collections.abc import Iterable


def check_iter_flatten() -> None:
    nested = (
        Range(3)
        .iter()
        .map(
            lambda x: (
                Range(x)
                .iter()
                .map(lambda y: Range(y).iter().map(lambda z: Range(z).pipe(list)))
            )
        )
    )
    _ = assert_type(nested, PyoIterator[PyoIterator[PyoIterator[list[int]]]])
    one = assert_type(nested.flatten(), PyoIterator[PyoIterator[list[int]]])
    two = assert_type(one.flatten(), PyoIterator[list[int]])
    ok = assert_type(two.flatten(), PyoIterator[int])
    # Expected to fail
    _fail = ok.flatten()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue, reportUnknownVariableType]


def check_chain_covariance[T, S](base: Iterable[T], *others: Iterable[S]) -> None:
    _ = assert_type(Iter(base).chain(*others), PyoIterator[T | S])
    _ = assert_type(itertools.chain(base, *others), itertools.chain[T | S])
