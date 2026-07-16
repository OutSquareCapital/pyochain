"""This modules contains tests that for the most part have been adapted from the CPython test suite for `heapq` functions.

Most modifications entails the conversion from `unittest` to `pytest`, the OOP shift (from functional paradigm to class-based), and the use of `HeapMin` and `HeapMax` instead of the `heapq` functions.

The original lives at:

https://github.com/python/cpython/blob/main/Lib/test/test_heapq.py
"""

from __future__ import annotations

import random
from itertools import chain
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Never, Self, final, override

import pytest

from pyochain import Vec
from pyochain.collections import Heap, HeapMax, HeapMin

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from types import NotImplementedType

    from _typeshed import SupportsRichComparison


def test_push_pop() -> None:
    # 1) Push 256 random numbers and pop them off, verifying all's OK.
    heap = HeapMin[float]([])
    data: list[float] = []
    check_invariant(heap)
    for _i in range(256):
        item = random.random()
        data.append(item)
        heap.push(item)
        check_invariant(heap)
    results: list[float] = []
    while heap:
        item = heap.pop()
        check_invariant(heap)
        results.append(item)
    data_sorted = data.copy()
    data_sorted.sort()
    assert data_sorted == results
    # 2) Check that the invariant holds for a sorted array
    check_invariant(results)

    with pytest.raises(TypeError):
        HeapMin[int]([]).push()  # pyright: ignore[reportCallIssue]


def test_max_push_pop() -> None:
    # 1) Push 256 random numbers and pop them off, verifying all's OK.
    heap = HeapMax[float]([])
    data: list[float] = []
    check_max_invariant(heap)
    for _ in range(256):
        item = random.random()
        data.append(item)
        heap.push(item)
        check_max_invariant(heap)
    results: list[float] = []
    while heap:
        item = heap.pop()
        check_max_invariant(heap)
        results.append(item)
    data_sorted = data.copy()
    data_sorted.sort(reverse=True)

    assert data_sorted == results
    # 2) Check that the invariant holds for a sorted array
    check_max_invariant(results)
    with pytest.raises(TypeError):
        HeapMax[int]([]).push()  # pyright: ignore[reportCallIssue]


def test_heapify() -> None:
    for size in [*list(range(30)), 20000]:
        heap = HeapMin([random.random() for _ in range(size)])
        check_invariant(heap)


def check_invariant(heap: Sequence[float]) -> None:
    # Check the heap invariant.
    for pos, item in enumerate(heap):
        if pos:  # pos 0 has no parent
            parentpos = (pos - 1) >> 1
            assert heap[parentpos] <= item


def test_heapify_max() -> None:
    for size in [*list(range(30)), 20000]:
        heap = HeapMax([random.random() for _ in range(size)])
        check_max_invariant(heap)


def check_max_invariant[T: SupportsRichComparison](heap: Sequence[T]) -> None:
    for pos, item in enumerate(heap[1:], start=1):
        parentpos = (pos - 1) >> 1
        assert heap[parentpos] >= item  # pyright: ignore[reportOperatorIssue]


def test_naive_nbest() -> None:
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMin[int]([])
    for item in data:
        _ = heap.push(item)
        if len(heap) > 10:
            _ = heap.pop()
    assert heap.iter().sort() == sorted(data)[-10:]


def test_nbest() -> None:
    # Less-naive "N-best" algorithm, much faster (if len(data) is big
    # enough <wink>) than sorting all of data.
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMin[int](data[:10])
    for item in data[10:]:
        if item > heap[0]:  # this gets rarer the longer we run
            _ = heap.replace(item)
    assert list(heapiter(heap)) == sorted(data)[-10:]
    with pytest.raises(IndexError):
        _ = HeapMin[int]([]).replace(None)  # pyright: ignore[reportArgumentType]


def test_nbest_maxheap() -> None:
    # With a max heap instead of a min heap, the "N-best" algorithm can
    # go even faster still via heapify'ing all of data (linear time), then
    # doing 10 heappops (10 log-time steps).
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMax(data.copy())
    result = [heap.pop() for _ in range(10)]
    result.reverse()
    assert result == sorted(data)[-10:]


def test_nbest_with_pushpop() -> None:
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMin(data[:10])
    for item in data[10:]:
        _ = heap.push_pop(item)
    assert list(heapiter(heap)) == sorted(data)[-10:]
    assert HeapMin[str]([]).push_pop("x") == "x"


def heapiter[T: SupportsRichComparison](heap: HeapMin[T]) -> Iterator[T]:
    # An iterator returning a heap's elements, smallest-first.
    try:
        while 1:
            yield heap.pop()
    except IndexError:
        pass


def test_naive_nworst() -> None:
    # Max-heap variant of "test_naive_nbest"
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMax[int]([])
    for item in data:
        heap.push(item)
        if heap.len() > 10:
            _ = heap.pop()
    assert heap.iter().sort() == sorted(data)[:10]


def test_nworst() -> None:
    # Max-heap variant of "test_nbest"
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMax(data[:10])
    for item in data[10:]:
        if item < heap[0]:  # this gets rarer the longer we run
            _ = heap.replace(item)
    expected = sorted(data, reverse=True)[-10:]
    assert list(heapiter_max(heap)) == expected
    with pytest.raises(IndexError):
        _ = HeapMax[int]([]).replace(None)  # pyright: ignore[reportArgumentType]


def test_nworst_minheap() -> None:
    # Min-heap variant of "test_nbest_maxheap"
    data = [random.randrange(2000) for _ in range(1000)]
    heap = HeapMin(data.copy())
    result = [heap.pop() for _ in range(10)]
    result.reverse()
    expected = sorted(data, reverse=True)[-10:]
    assert result == expected


def test_nworst_with_pushpop() -> None:
    # Max-heap variant of "test_nbest_with_pushpop"
    data = Vec([random.randrange(2000) for _ in range(1000)])
    heap = HeapMax(data[:10])
    for item in data[10:]:
        _ = heap.push_pop(item)
    expected = data.iter().sort(reverse=True)[-10:]
    assert list(heapiter_max(heap)) == expected
    assert HeapMax[str]([]).push_pop("x") == "x"


def heapiter_max[T: SupportsRichComparison](heap: HeapMax[T]) -> Iterator[T]:
    # An iterator returning a max-heap's elements, largest-first.
    try:
        while 1:
            yield heap.pop()
    except IndexError:
        pass


def test_pushpop() -> None:
    h = HeapMin[int]([])
    x = h.push_pop(10)
    assert (h, x) == ([], 10)

    h = HeapMin[float]([10])
    x = h.push_pop(10.0)
    assert (h, x) == ([10], 10.0)
    assert type(h[0]) is int
    assert type(x) is float

    h = HeapMin([10])
    x = h.push_pop(9)
    assert (h, x) == ([10], 9)

    h = HeapMin([10])
    x = h.push_pop(11)
    assert (h, x) == ([11], 10)


def test_pushpop_max() -> None:
    h = HeapMax[int]([])
    x = h.push_pop(10)
    assert (h, x) == ([], 10)

    h = HeapMax[float]([10])
    x = h.push_pop(10.0)
    assert (h, x) == ([10], 10.0)
    assert isinstance(h[0], int)
    assert isinstance(x, float)

    h = HeapMax([10])
    x = h.push_pop(11)
    assert (h, x) == ([10], 11)

    h = HeapMax([10])
    x = h.push_pop(9)
    assert (h, x) == ([9], 10)


def test_heappop_max() -> None:
    # heapop_max has an optimization for one-item lists which isn't
    # covered in other tests, so test that case explicitly here
    h = HeapMax([3, 2])
    assert h.pop() == 3
    assert h.pop() == 2


def test_heapsort() -> None:
    # Exercise everything with repeated heapsort checks
    for trial in range(100):
        size = random.randrange(50)
        data = [random.randrange(25) for _ in range(size)]
        if trial & 1:  # Half of the time, use heapify
            heap: HeapMin[int] = HeapMin(data.copy())
        else:  # The rest of the time, use push
            heap = HeapMin([])
            for item in data:
                _ = heap.push(item)
        heap_sorted = [heap.pop() for _ in range(size)]
        assert heap_sorted == sorted(data)


def test_heapsort_max() -> None:
    for trial in range(100):
        size = random.randrange(50)
        data = [random.randrange(25) for _ in range(size)]
        if trial & 1:  # Half of the time, use heapify_max
            heap: HeapMax[int] = HeapMax(data.copy())
        else:  # The rest of the time, use push_max
            heap = HeapMax([])
            for item in data:
                _ = heap.push(item)
        heap_sorted = [heap.pop() for _ in range(size)]
        assert heap_sorted == sorted(data, reverse=True)


def test_merge() -> None:
    inputs: list[list[tuple[str, int]]] = []
    for _i in range(random.randrange(25)):
        row: list[tuple[str, int]] = []
        for _j in range(random.randrange(100)):
            tup = random.choice("ABC"), random.randrange(-500, 500)
            row.append(tup)
        inputs.append(row)

    for key in [None, itemgetter(0), itemgetter(1), itemgetter(1, 0)]:
        for reverse in [False, True]:
            base = HeapMin[tuple[str, int]]([])
            seqs = [sorted(seq, key=key, reverse=reverse) for seq in inputs]
            assert sorted(chain(*inputs), key=key, reverse=reverse) == list(
                base.merge(*seqs, key=key, reverse=reverse)
            )
            assert list(base.merge()) == []


def test_empty_merges() -> None:
    # Merging two empty lists (with or without a key) should produce
    # another empty list.
    assert list(HeapMin[int]([]).merge([])) == []
    assert list(HeapMin[int]([]).merge([], key=lambda _: 6)) == []


def test_merge_does_not_suppress_index_error() -> None:
    # Issue 19018: Heapq.merge suppresses IndexError from user generator
    def iterable() -> Iterator[int]:
        s = list(range(10))
        for i in range(20):
            yield s[i]  # IndexError when i > 10

    with pytest.raises(IndexError):
        _ = list(HeapMin[int]([]).merge(iterable(), iterable()))


def test_merge_stability() -> None:
    class Int(int):
        pass

    base = HeapMin[Int]([])
    others: list[list[Int]] = [[], [], [], []]
    for _i in range(20000):
        stream = random.randrange(4)
        x = random.randrange(500)
        obj = Int(x)
        obj.pair = (x, stream)  # pyright: ignore[reportAttributeAccessIssue]
        others[stream].append(obj)
    for stream in others:
        stream.sort()
    result: list[tuple[int, int]] = [i.pair for i in base.merge(*others)]  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
    assert result == sorted(result)


def test_nsmallest() -> None:
    data = HeapMin([(random.randrange(2000), i) for i in range(1000)])
    for f in (None, _op_for_n):
        for n in (0, 1, 2, 10, 100, 400, 999, 1000, 1100):
            assert list(data.n_smallest(n)) == sorted(data)[:n]
            assert list(data.n_smallest(n, key=f)) == sorted(data, key=f)[:n]


def test_nlargest() -> None:
    data = HeapMax([(random.randrange(2000), i) for i in range(1000)])
    for f in (None, _op_for_n):
        for n in (0, 1, 2, 10, 100, 400, 999, 1000, 1100):
            assert list(data.n_largest(n)) == sorted(data, reverse=True)[:n]
            assert (
                list(data.n_largest(n, key=f)) == sorted(data, key=f, reverse=True)[:n]
            )


def _op_for_n(x: Sequence[int]) -> int:
    return x[0] * 547 % 2000


def test_comparison_operator() -> None:
    # Issue 3051: Make sure heapq works with both __lt__
    # For python 3.0, __le__ alone is not enough
    class _HasX:  # noqa: B903
        def __init__(self, x: float) -> None:
            self.x: float = x

    class ImplLT(_HasX):
        def __lt__(self, other: Self) -> bool:
            return self.x > other.x

    class ImplLE(_HasX):
        def __le__(self, other: Self) -> bool:
            return self.x >= other.x

    def hsort[T: SupportsRichComparison](
        data: list[float], comp: Callable[[float], _HasX]
    ) -> list[float]:
        heap = HeapMin[_HasX]([comp(x) for x in data])  # pyright: ignore[reportInvalidTypeArguments]
        return [heap.pop().x for _ in range(len(data))]

    data = [random.random() for _ in range(100)]
    target = sorted(data, reverse=True)
    assert hsort(data, ImplLT) == target
    with pytest.raises(TypeError):
        _ = hsort(data, ImplLE)


# ==============================================================================


class LenOnly:
    """Dummy sequence class defining __len__ but not __getitem__."""

    def __len__(self) -> int:
        return 10


@final
class CmpErr:  # noqa: PLW1641
    """Dummy element that always raises an error during comparison."""

    @override
    def __eq__(self, other: object) -> Never:
        raise ZeroDivisionError

    __ne__ = __lt__ = __le__ = __gt__ = __ge__ = __eq__


class ImplGetItem:
    """Sequence using __getitem__."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn

    def __getitem__(self, i: int) -> int:
        return self.seqn[i]


class ImplIterator:
    """Sequence using iterator protocol."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn
        self.i: int = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> int:
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class ImplGenerator:
    """Sequence using iterator protocol defined with a generator."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn
        self.i: int = 0

    def __iter__(self) -> Iterator[int]:
        yield from self.seqn


class MissGetItemAndIter:
    """Missing __getitem__ and __iter__."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn
        self.i: int = 0

    def __next__(self) -> int:
        if self.i >= len(self.seqn):
            raise StopIteration
        v = self.seqn[self.i]
        self.i += 1
        return v


class MissNext:
    """Iterator missing __next__()."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn
        self.i: int = 0

    def __iter__(self) -> Self:
        return self


class PropagateException:
    """Test propagation of exceptions."""

    def __init__(self, seqn: Heap[int]) -> None:
        self.seqn: Heap[int] = seqn
        self.i: int = 0

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> None:
        _ = 3 // 0


class RaiseImmediateStop:
    """Test immediate stop."""

    def __init__(self, seqn: Heap[int]) -> None:
        pass

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> Never:
        raise StopIteration


def multiple_iterators(seqn: Heap[int]) -> Iterator[int]:
    """Test multiple tiers of iterators."""
    return chain(map(lambda x: x, reg_generator(ImplGenerator(ImplGetItem(seqn)))))  # pyright: ignore[ reportArgumentType]


def reg_generator(seqn: Heap[int]) -> Iterator[int]:
    """Regular generator."""
    yield from seqn


class SideEffectLT[T: SupportsRichComparison]:
    def __init__(self, value: int, heap: Heap[T]) -> None:
        self.value: int = value
        self.heap: Heap[T] = heap

    def __lt__(self, other: Self) -> bool:
        self.heap[:] = []
        return self.value < other.value


# TestErrorHandling:


def test_non_sequence() -> None:
    for f in (
        HeapMin[int].__init__,
        HeapMin[int].pop,
        HeapMax[int].__init__,
        HeapMax[int].pop,
    ):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(10)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]
    for f in (
        HeapMin[int].push,
        HeapMin[int].replace,
        HeapMax[int].push,
        HeapMax[int].replace,
        HeapMin[int].n_largest,
        HeapMin[int].n_smallest,
    ):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(10, 10)  # pyright: ignore[reportArgumentType]


def test_len_only() -> None:
    for f in (
        HeapMin[int].__init__,
        HeapMin[int].pop,
        HeapMax[int].__init__,
        HeapMax[int].pop,
    ):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(LenOnly())  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]

    for f in (
        HeapMin[int].push,
        HeapMin[int].replace,
        HeapMax[int].push,
        HeapMax[int].replace,
    ):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(LenOnly(), 10)  # pyright: ignore[reportArgumentType]
    for f in (HeapMin[int].n_largest, HeapMin[int].n_smallest):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(2, LenOnly())  # pyright: ignore[reportArgumentType]


def test_cmp_err() -> None:
    data = [CmpErr(), CmpErr(), CmpErr()]
    heapmin = HeapMin[CmpErr].from_ref(data)
    heapmax = HeapMax[CmpErr].from_ref(data.copy())
    with pytest.raises(ZeroDivisionError):
        HeapMin[CmpErr].__init__(heapmin, heapmin.inner)
    with pytest.raises(ZeroDivisionError):
        _ = heapmin.pop()
    with pytest.raises(ZeroDivisionError):
        _ = heapmin.push(10)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ZeroDivisionError):
        _ = heapmin.replace(10)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ZeroDivisionError):
        _ = heapmax.push(10)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ZeroDivisionError):
        _ = heapmax.replace(10)  # pyright: ignore[reportArgumentType]
    for f in (Heap[CmpErr].n_largest, Heap[CmpErr].n_smallest):
        with pytest.raises(ZeroDivisionError):
            _ = f(heapmin, 2)
        with pytest.raises(ZeroDivisionError):
            _ = f(heapmax, 2)


def test_arg_parsing() -> None:
    for f in (
        HeapMin[int].__init__,
        HeapMin[int].pop,
        HeapMin[int].push,
        HeapMin[int].replace,
        HeapMax[int].__init__,
        HeapMax[int].pop,
        HeapMax[int].push,
        HeapMax[int].replace,
        HeapMin[int].n_largest,
        HeapMin[int].n_smallest,
    ):
        with pytest.raises((TypeError, AttributeError)):
            _ = f(10)  # pyright: ignore[reportCallIssue, reportArgumentType, reportUnknownVariableType]


def test_iterable_args() -> None:
    for f in (HeapMin[float].n_largest, HeapMin[float].n_smallest):
        for s in ("123", "", range(1000), (1, 1.2), range(2000, 2200, 5)):
            for g in (
                ImplGetItem,
                ImplIterator,
                ImplGenerator,
                multiple_iterators,
                reg_generator,
            ):
                assert list(f(HeapMin[float].from_ref(g(s)), 2)) == list(  # pyright: ignore[reportArgumentType]
                    f(HeapMin[float].from_ref(s), 2)  # pyright: ignore[reportArgumentType]
                )
            assert list(f(HeapMin[float].from_ref(RaiseImmediateStop(s)), 2)) == []  # pyright: ignore[reportArgumentType]
            with pytest.raises(TypeError):
                _ = f(HeapMin[float].from_ref(MissGetItemAndIter(s)), 2)  # pyright: ignore[reportArgumentType]
            with pytest.raises(TypeError):
                _ = f(HeapMin[float].from_ref(MissNext(s)), 2)  # pyright: ignore[reportArgumentType]
            with pytest.raises(ZeroDivisionError):
                _ = f(HeapMin[float].from_ref(PropagateException(s)), 2)  # pyright: ignore[reportArgumentType]


# Issue #17278: the heap may change size while it's being walked.


def test_push_mutating_heap() -> None:
    heap = HeapMin[Any]([])
    heap.extend(SideEffectLT(i, heap) for i in range(200))
    # Python version raises IndexError, C version RuntimeError
    with pytest.raises((IndexError, RuntimeError)):
        heap.push(SideEffectLT(5, heap))
    heap = HeapMax[Any]([])
    heap.extend(SideEffectLT(i, heap) for i in range(200))
    with pytest.raises((IndexError, RuntimeError)):
        heap.push(SideEffectLT(5, heap))


def test_heappop_mutating_heap() -> None:
    heap = HeapMin[Any]([])
    heap.extend(SideEffectLT(i, heap) for i in range(200))
    # Python version raises IndexError, C version RuntimeError
    with pytest.raises((IndexError, RuntimeError)):
        _ = heap.pop()  # pyright: ignore[reportAny]
    heap = HeapMax[Any]([])
    heap.extend(SideEffectLT(i, heap) for i in range(200))
    with pytest.raises((IndexError, RuntimeError)):
        _ = heap.pop()  # pyright: ignore[reportAny]


def test_comparison_operator_modifying_heap() -> None:
    # See bpo-39421: Strong references need to be taken
    # when comparing objects as they can alter the heap
    class EvilClass(int):
        @override
        def __lt__(self, o: object) -> NotImplementedType:
            heap.clear()
            return NotImplemented

    heap = HeapMin[Any]([])
    heap.push(EvilClass(0))
    with pytest.raises(IndexError):
        _ = heap.push_pop(1)  # pyright: ignore[reportAny]


def test_comparison_operator_modifying_heap_two_heaps() -> None:

    class MutList1(int):
        @override
        def __lt__(self, o: object) -> NotImplementedType:
            list2.clear()
            return NotImplemented

    class MutList2(int):
        @override
        def __lt__(self, o: object) -> NotImplementedType:
            list1.clear()
            return NotImplemented

    list1 = HeapMin[int]([])
    list2 = HeapMin[int]([])

    list1.push(MutList1(0))
    list2.push(MutList2(0))
    with pytest.raises((IndexError, RuntimeError)):
        list1.push(MutList2(1))
    with pytest.raises((IndexError, RuntimeError)):
        list2.push(MutList1(1))

    list1 = HeapMax[int]([])
    list2 = HeapMax[int]([])

    list1.push(MutList1(0))
    list2.push(MutList2(0))
    list1.push(MutList2(1))
    list2.push(MutList1(1))

    with pytest.raises((IndexError, RuntimeError)):
        list1.push(MutList2(1))
    with pytest.raises((IndexError, RuntimeError)):
        list2.push(MutList1(1))
