"""Tests for the internal Range wrapper."""

from pyochain import Range


def test_range_matches_builtin_range_iteration_and_length() -> None:
    """Range should delegate iteration and length to the builtin range."""
    wrapped = Range(1, 10, 2)
    expected = range(1, 10, 2)

    assert tuple(wrapped) == tuple(expected)
    assert len(wrapped) == len(expected)


def test_range_supports_indexing_with_positive_and_negative_indices() -> None:
    """Range should expose builtin range indexing semantics."""
    wrapped = Range(3, 15, 3)
    expected = range(3, 15, 3)

    assert wrapped[0] == expected[0]
    assert wrapped[-1] == expected[-1]


def test_range_supports_slicing() -> None:
    """Range slices should match builtin range slices."""
    wrapped = Range(2, 20, 3)
    expected = range(2, 20, 3)

    assert wrapped[1:4] == expected[1:4]
    assert wrapped[::-1] == expected[::-1]


def test_range_supports_negative_steps() -> None:
    """Range should preserve builtin descending range behavior."""
    wrapped = Range(10, 0, -2)
    expected = range(10, 0, -2)

    assert tuple(wrapped) == tuple(expected)
    assert wrapped[2] == expected[2]
