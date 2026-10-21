from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from random import Random
from typing import TYPE_CHECKING, Final, override

import pytest
from sortedcontainers import SortedDict as PySortedDict
from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList
from sortedcontainers import SortedSet as PySortedSet

from pyochain import Range, Seq, Vec
from pyochain.collections import SortedDict, SortedKeyList, SortedList, SortedSet

from .._utils import SIZES

if TYPE_CHECKING:
    from collections.abc import Callable

    from _pytest.mark import MarkDecorator, ParameterSet

    from .._utils import BenchFixture
SEED: Final[int] = 0
RANDOM: Final[Random] = Random(SEED)
SC_SIZES: Final[MarkDecorator] = pytest.mark.parametrize("size", SIZES[1:])
"""`sortedcontainers` original benchs don't have a size of 10, so we skip it when the scaling (e.g `.step_by(10)`) cause issues."""
type List = SortedList[int] | PySortedList[int]
type Set = SortedSet[int] | PySortedSet[int]
type SortedSequence = List | Set
type Dict = SortedDict[int, int] | PySortedDict[int, int]
type SortedCollection = SortedSequence | Dict
type KeyList = SortedKeyList[int, int] | PySortedKeyList[int, int]
type InitFn[E, L] = Callable[[Iterable[E]], L]
type InitDict[D] = InitFn[tuple[int, int], D]
type KeyFn = Callable[[int], int]
type SetupOut[L] = tuple[tuple[L], dict[str, object]]


def _make_params(pyochain: type, sortedcontainers: type) -> MarkDecorator:
    args = (
        pytest.param(sortedcontainers, id="sortedcontainers"),
        pytest.param(pyochain, id="pyochain"),
    )
    return pytest.mark.parametrize("cls", args)


SEQ_PARAMS: Final[MarkDecorator] = pytest.mark.parametrize(
    "cls",
    (
        pytest.param(PySortedList, id="sortedcontainers-list"),
        pytest.param(SortedList, id="pyochain-list"),
        pytest.param(PySortedSet, id="sortedcontainers-set"),
        pytest.param(SortedSet, id="pyochain-set"),
    ),
)
CLS_PARAMS: Final[MarkDecorator] = _make_params(PySortedList, SortedList)
CLS_DICT_PARAMS: Final[MarkDecorator] = _make_params(PySortedDict, SortedDict)
CLS_SET_PARAMS: Final[MarkDecorator] = _make_params(PySortedSet, SortedSet)


@dataclass(slots=True)
class _BaseConfig[C](ABC):
    bench: BenchFixture
    cls: type[C]
    size: int

    @abstractmethod
    def new_from_size(self) -> C: ...
    def run_pedantic(self, f: Callable[[C], object], *, filled: bool = True) -> None:
        min_rounds = 20
        rounds = max(
            min_rounds, int(50 * math.sqrt(SIZES.last() // max(self.size, 4_000)))
        )
        warmup_rounds = rounds // min_rounds
        length = calls_nb(SIZES.last()) // calls_nb(self.size)

        def setup() -> SetupOut[Seq[C]]:
            f = self._make(filled)
            data = Range(length).iter().map(lambda _: f()).collect(Seq[C])
            return (data,), {}

        def target(data: Seq[C]) -> None:
            data.iter().for_each(f)

        self.bench.pedantic(
            target, setup=setup, rounds=rounds, warmup_rounds=warmup_rounds
        )

    def _make(self, filled: bool) -> Callable[[], C]:
        if filled:
            return self.new_from_size
        else:
            return self.cls


@dataclass(slots=True)
class SeqConfig[S: SortedSequence](_BaseConfig[S]):
    """Config for `Sequence` collections (List, Set)."""

    def run_workload(self, op: Callable[[S, int, int], object]) -> None:
        values = (
            shuffled(self.size, 10)
            .tap(lambda _: RANDOM.seed(SEED))
            .iter()
            .map(lambda val: (val, RANDOM.randrange(start=100)))
            .collect(Seq)
        )

        def f(sl: S) -> None:
            for value, pos in values:
                _ = op(sl, value, pos)

        self.run_pedantic(f, filled=True)

    @override
    def new_from_size(self) -> S:
        return self.cls(Range(self.size))


@dataclass(slots=True)
class DictConfig[D: Dict](_BaseConfig[D]):
    @override
    def new_from_size(self) -> D:
        return self.cls(Range(self.size).iter().map(item_to_pair))


def item_to_pair(item: int) -> tuple[int, int]:
    return item, -item


@cache
def shuffled(size: int, step: int = 1) -> Vec[int]:
    return Vec.wrap(RANDOM.sample(range(size), size // step))


def calls_nb(size: int) -> int:
    return min(size // 4, 1_000)


class Stop(Enum):
    Tiny = auto()
    Small = auto()
    Medium = auto()
    Large = auto()

    def fn(self, size: int) -> int:
        match self:
            case self.Tiny:
                return min(10, size)
            case self.Small:
                return size // 10
            case self.Medium:
                return size * 5 // 10
            case self.Large:
                return size * 9 // 10

    def as_param(self) -> ParameterSet:
        return pytest.param(self, id=self.name)
