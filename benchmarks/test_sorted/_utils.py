from __future__ import annotations

import math
from collections.abc import Iterable
from functools import cache
from random import Random
from typing import TYPE_CHECKING, Final, TypeIs

import pytest
import sortedcontainers as sc

from pyochain import Range, Seq, Vec
from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet

from .._utils import SIZES, Lib

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import MarkDecorator, ParameterSet

    from pyochain.abc import PyoIterable

    from .._utils import BenchFixture
SEED: Final[int] = 0
RANDOM: Final[Random] = Random(SEED)
SC_SIZES: Final[MarkDecorator] = pytest.mark.parametrize("size", SIZES[1:])
"""`sortedcontainers` original benchs don't have a size of 10, so we skip it when the scaling (e.g `.step_by(10)`) cause issues."""
type List = SortedList[int] | sc.SortedList[int]
type Set = SortedSet[int] | sc.SortedSet[int]
type SortedSequence = List | Set
type Dict = SortedDict[int, int] | sc.SortedDict[int, int]
type SortedCollection = SortedSequence | Dict
type KeyList = SortedKeyList[int, int] | sc.SortedKeyList[int, int]
type InitFn[E, L] = Callable[[Iterable[E]], L]
type InitDict[D] = InitFn[tuple[int, int], D]
type KeyFn = Callable[[int], int]
type SetupOut[L] = tuple[tuple[L], dict[str, object]]
type AddFn = Callable[[List, int], None]


def make_params(pyochain: type, sortedcontainers: type) -> Seq[ParameterSet]:
    return Seq(
        Lib.Pyochain.to_param(pyochain), Lib.SortedContainers.to_param(sortedcontainers)
    )


_LISTS = make_params(SortedList, sc.SortedList)
_SETS = make_params(SortedSet, sc.SortedSet)
_SEQS = _LISTS + _SETS
_DICTS = make_params(SortedDict, sc.SortedDict)
LST_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize("cls", _LISTS)
SET_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize("cls", _SETS)
DICT_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize("cls", _DICTS)
SEQ_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize("cls", _SEQS)
COLLECTION_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize("cls", _SEQS + _DICTS)


def run_pedantic[C: SortedCollection](
    bench: BenchFixture,
    cls: type[C],
    size: int,
    f: Callable[[C], object],
    *,
    empty: bool = False,
) -> None:
    min_rounds = 20
    rounds = max(min_rounds, int(50 * math.sqrt(SIZES.last() // max(size, 4_000))))
    warmup_rounds = rounds // min_rounds

    def setup() -> SetupOut[Seq[C]]:
        values = Range(size if not empty else 0)
        data = (
            Range(_calls_nb(SIZES.last()) // _calls_nb(size))
            .iter()
            .map(lambda _: into_sorted(values, cls))
            .collect(Seq)
        )
        return (data,), {}

    def target(data: Seq[C]) -> None:
        data.iter().for_each(f)

    bench.pedantic(target, setup=setup, rounds=rounds, warmup_rounds=warmup_rounds)


def _calls_nb(size: int) -> int:
    return min(size // 4, 1_000)


def into_sorted[C: SortedCollection](values: PyoIterable[int], cls: type[C]) -> C:
    if is_dict(cls):
        return cls(values.iter().map(_item_to_pair))  # pyright: ignore[reportReturnType]
    else:
        return cls(values)  # pyright: ignore[reportReturnType]


def is_dict(cls: type[SortedCollection]) -> TypeIs[type[Dict]]:
    return cls is SortedDict or cls is sc.SortedDict


@cache
def shuffled_pairs(size: int, step: int = 1) -> Vec[tuple[int, int]]:
    return shuffled(size, step).iter().map(_item_to_pair).collect(Vec)


@cache
def shuffled(size: int, step: int = 1) -> Vec[int]:
    return Vec.wrap(RANDOM.sample(range(size), size // step))


def _item_to_pair(item: int) -> tuple[int, int]:
    return item, -item


def add_fn(cls: type[SortedSequence]) -> Callable[[SortedSequence, int], None]:
    if cls is SortedList:
        return SortedList[int].append  # pyright: ignore[reportReturnType]
    else:
        return cls.add  # pyright: ignore[reportReturnType]
