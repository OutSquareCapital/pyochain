from collections.abc import Callable
from typing import Final

import pytest

from pyochain import Vec

BASE: Final[list[int]] = [1, 2, 3, 4, 5]
EMPTY: Final[list[int]] = []
type Predicate = Callable[[int], bool]


@pytest.mark.parametrize(
    ("start", "stop", "expected_next", "expected_list"),
    (
        (1, 4, 2, [1, 5]),
        (None, None, 1, EMPTY),
        (0, 2, 1, [3, 4, 5]),
        (2, None, 3, [1, 2]),
        (None, 3, 1, [4, 5]),
    ),
)
def test_drain_partial_consumption(
    start: int | None, stop: int | None, expected_next: int, expected_list: list[int]
) -> None:
    v = Vec(BASE)
    drain_iter = v.drain(start, stop)
    assert drain_iter.next().unwrap() == expected_next
    del drain_iter
    assert v == expected_list


def test_drain_no_consumption_gc() -> None:
    v = Vec(BASE)
    drain_iter = v.drain(1, 3)
    del drain_iter
    assert v == [1, 4, 5]


def test_drain_full_consumption() -> None:
    v = Vec(BASE)
    drained = v.drain(1, 2).collect(Vec)
    assert drained == [2]
    assert v == [1, 3, 4, 5]


RETAIN_PARAMS: Final[list[tuple[Predicate, list[int]]]] = [
    (lambda x: x % 2 == 0, [2, 4]),
    (lambda x: x > 10, EMPTY),
    (lambda x: x > 0, BASE),
]


@pytest.mark.parametrize(("pred", "expected"), RETAIN_PARAMS)
def test_retain(pred: Predicate, expected: list[int]) -> None:
    v = Vec(BASE)
    v.retain(pred)
    assert v == expected


@pytest.mark.parametrize(
    ("n", "expected"), ((1, [1]), (0, []), (10, BASE), (3, [1, 2, 3]))
)
def test_truncate(n: int, expected: list[object]) -> None:
    v = Vec(BASE)
    v.truncate(n)
    assert v == expected


EXTRACT_IF_PARAMS: Final[list[tuple[Predicate, int, int, list[int], list[int]]]] = [
    (lambda x: x % 2 == 0, 0, 5, [2, 4], [1, 3, 5]),
    (lambda x: x > 10, 0, 5, EMPTY, BASE),
    (lambda x: x > 0, 0, 5, BASE, EMPTY),
    (lambda x: x % 2 == 0, 1, 4, [2, 4], [1, 3, 5]),
]


@pytest.mark.parametrize(
    ("pred", "start", "stop", "extracted_expected", "original_expected"),
    EXTRACT_IF_PARAMS,
)
def test_extract_if(
    pred: Predicate,
    start: int,
    stop: int,
    extracted_expected: list[int],
    original_expected: list[int],
) -> None:
    v = Vec(BASE)
    assert v.extract_if(pred, start, stop).collect(Vec) == extracted_expected
    assert v == original_expected
