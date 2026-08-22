import pytest

from pyochain import Range


def test_range_single_arg() -> None:
    r = Range(5)
    assert list(r) == [0, 1, 2, 3, 4]


def test_range_two_args() -> None:
    r = Range(2, 6)
    assert list(r) == [2, 3, 4, 5]


def test_range_three_args() -> None:
    r = Range(1, 10, 2)
    assert list(r) == [1, 3, 5, 7, 9]


def test_range_invalid_args() -> None:
    with pytest.raises(TypeError, match="range expected at most 3 arguments"):
        Range(1, 2, 3, 4)
