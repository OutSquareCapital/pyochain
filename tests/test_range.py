import pytest

from pyochain import Range


@pytest.mark.parametrize(
    ("args", "expected"),
    (
        ((5,), [0, 1, 2, 3, 4]),
        ((2, 6), [2, 3, 4, 5]),
        ((1, 10, 2), [1, 3, 5, 7, 9]),
    ),
)
def test_range_args(args: tuple[int, ...], expected: list[int]) -> None:
    assert list(Range(*args)) == expected


def test_range_invalid_args() -> None:
    with pytest.raises(TypeError, match="range expected at most 3 arguments"):
        Range(1, 2, 3, 4)
