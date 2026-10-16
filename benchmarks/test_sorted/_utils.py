from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from random import Random
from typing import TYPE_CHECKING, Final

from sortedcontainers import SortedList as PySortedList
from sortedcontainers import SortedListWithKey as PySortedKeyList

from benchmarks._utils import SIZES
from pyochain import Range, Seq, Vec
from pyochain.collections import SortedKeyList, SortedList

if TYPE_CHECKING:
    from collections.abc import Callable

    from .._utils import BenchFixture

RANDOM: Final[Random] = Random(0)

type List = SortedList[int] | PySortedList[int]
type KeyList = type[SortedKeyList[int, int] | PySortedKeyList[int, int]]

type SetupOut[L] = tuple[tuple[L], dict[str, object]]


@dataclass(slots=True)
class BenchConfig[L: List]:
    bench: BenchFixture
    cls: type[L]
    size: int

    def run_workload(self, op: Callable[[List, int, int], object]) -> None:
        values = shuffled(self.size).iter().step_by(10).collect(Seq)

        def f(sl: List) -> None:
            for value in values:
                _ = op(sl, value, RANDOM.randrange(100))

        self.run_pedantic(f)

    def run_pedantic(self, f: Callable[[L], object]) -> None:
        reps = Range(calls_nb(SIZES.last()) // calls_nb(self.size))
        rounds = max(20, int(50 * math.sqrt(SIZES.last() // max(self.size, 4_000))))

        def run(lists: Seq[L]) -> None:
            for sl in lists:
                _ = f(sl)

        def setup() -> SetupOut[Seq[L]]:
            data = reps.iter().map(lambda _: self.cls(range(self.size))).collect(Seq)
            return (data,), {}

        self.bench.pedantic(run, setup=setup, rounds=rounds, warmup_rounds=rounds // 20)


@cache
def shuffled(size: int) -> Vec[int]:
    return Vec.wrap(RANDOM.sample(range(size), size))


def calls_nb(size: int) -> int:
    return min(size // 4, 1_000)
