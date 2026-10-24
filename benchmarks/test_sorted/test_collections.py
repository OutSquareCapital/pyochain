from __future__ import annotations

import operator
from typing import TYPE_CHECKING

import pytest
import sortedcontainers as sc

from pyochain import Iter, Range, Seq
from pyochain.collections import SortedSet

from .._utils import SIZE_PARAMS, consume
from ._utils import (
    COLLECTION_PARAMS,
    RANDOM,
    SC_SIZES,
    SEED,
    SEQ_PARAMS,
    SortedCollection,
    SortedSequence,
    add_fn,
    into_sorted,
    is_dict,
    is_type_of,
    run_pedantic,
    shuffled,
    shuffled_pairs,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._utils import BenchFixture

type Cls = type[SortedCollection]
type FnFactory = Callable[[SortedCollection], Callable[[int], object]]


LOOKUP_FNS = (
    Iter[tuple[FnFactory, str]](
        (lambda lst: lst.__getitem__, "getitem"),
        (lambda lst: lst.__contains__, "contains"),
        (lambda lst: lst.index, "index"),
        (lambda lst: lst.bisect_right, "bisect_right"),
    )
    .map_star(lambda f, name: pytest.param(f, id=name))
    .collect(Seq)
)


@COLLECTION_PARAMS
@SIZE_PARAMS
def test_init[C: SortedCollection](
    benchmark: BenchFixture, cls: type[C], size: int
) -> None:
    if is_dict(cls):
        values = shuffled_pairs(size)
        init = lambda: cls(values)
    else:
        values = shuffled(size)
        init = lambda: cls(values)
    assert benchmark(init)


@COLLECTION_PARAMS
@SIZE_PARAMS
def test_iter(benchmark: BenchFixture, cls: Cls, size: int) -> None:
    benchmark(consume, into_sorted(Range(size), cls))


@COLLECTION_PARAMS
@SC_SIZES
def test_delitem(benchmark: BenchFixture, cls: Cls, size: int) -> None:
    if is_dict(cls):
        values = shuffled(size, 100)
    else:
        RANDOM.seed(SEED)
        values = (
            Range(size // 100)
            .iter()
            .map(lambda val: RANDOM.randrange(size - val))
            .collect(Seq)
        )
    run_pedantic(benchmark, cls, size, lambda x: values.iter().for_each(x.__delitem__))


@pytest.mark.parametrize("factory", LOOKUP_FNS)
@COLLECTION_PARAMS
@SC_SIZES
def test_lookup(
    benchmark: BenchFixture, cls: Cls, size: int, factory: FnFactory
) -> None:
    values = shuffled(size, 100)
    f = factory(into_sorted(Range(size), cls))
    benchmark(lambda: values.iter().for_each(f))


@SEQ_PARAMS
@SC_SIZES
def test_count(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    values = shuffled(size, 100)

    f = cls(Range(size)).count
    benchmark(lambda: values.iter().for_each(f))


@SEQ_PARAMS
@SC_SIZES
def test_remove(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    values = shuffled(size, 100)
    run_pedantic(benchmark, cls, size, lambda lst: values.iter().for_each(lst.remove))


@SEQ_PARAMS
@SC_SIZES
def test_pop(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    values = Range(size // 100)

    def f(lst: SortedSequence) -> None:
        values.iter().for_each(lambda _: lst.pop())

    run_pedantic(benchmark, cls, size, f)


@SEQ_PARAMS
@SC_SIZES
def test_add(benchmark: BenchFixture, cls: type[SortedSequence], size: int) -> None:
    add_method = add_fn(cls)
    values = shuffled(size, 100)
    if is_type_of(cls, sc.SortedSet) or is_type_of(cls, SortedSet):
        values = values.iter().map(operator.neg).collect(Seq)

    def f(lst: SortedSequence) -> None:
        for val in values:
            add_method(lst, val)

    run_pedantic(benchmark, cls, size, f)
