"""Tests for all_equal function."""

import pyochain as pc


def test_all_equal_with_identical_elements() -> None:
    """Test all_equal with all identical elements."""
    result = pc.Iter((5, 5, 5, 5)).all_equal(lambda x: x)
    assert result is True


def test_all_equal_with_different_elements() -> None:
    """Test all_equal with different elements."""
    result = pc.Iter((1, 2, 3, 4)).all_equal(lambda x: x)
    assert result is False


def test_all_equal_with_key_function() -> None:
    """Test all_equal with a key function that groups elements."""
    result = pc.Iter((2, 4, 6, 8)).all_equal(lambda x: x % 2)
    assert result is True


def test_all_equal_empty_sequence() -> None:
    """Test all_equal with empty sequence returns True."""
    result = pc.Iter[int].new().all_equal(lambda x: x)
    assert result is True


def test_all_equal_single_element() -> None:
    """Test all_equal with single element returns True."""
    result = pc.Iter.once(42).all_equal(lambda x: x)
    assert result is True
