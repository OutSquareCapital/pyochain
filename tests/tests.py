"""Tests for slot usage in pyochain classes."""

import pyochain as pc


def _check_slots(obj: object) -> bool:
    try:
        _x = obj.__dict__
        return False  # noqa: TRY300
    except AttributeError:
        return True


def test_slots() -> None:  # noqa: D103
    assert _check_slots(pc.Iter(()))
    assert _check_slots(pc.Seq(()))
    assert _check_slots(pc.Set(()))
    assert _check_slots(pc.SetMut(()))
    assert _check_slots(pc.Dict[str, str].new())
    assert _check_slots(pc.Some(42))
    assert _check_slots(pc.NoneOption())
    assert _check_slots(pc.Err[int, object](42))
    assert _check_slots(pc.Ok[int, object](42))


def test_drain_basic() -> None:
    """Test basic drain functionality with garbage collection."""
    v = pc.Vec([1, 2, 3, 4, 5])
    drain_iter = v.drain(1, 4)
    # Don't consume fully, let garbage collection handle remaining elements
    next(drain_iter)  # consume only first element (2)
    del drain_iter  # Trigger cleanup via __del__
    assert list(v) == [1, 5]


def test_drain_no_args_gc() -> None:
    """Test drain with no arguments - verify garbage collection cleanup."""
    v = pc.Vec([1, 2, 3, 4, 5])
    drain_iter = v.drain()
    next(drain_iter)  # consume first element only
    del drain_iter  # Remaining elements should be cleaned up
    assert list(v) == []


def test_drain_partial_consumption() -> None:
    """Test that partially consumed drain cleans up remaining elements via GC."""
    v = pc.Vec([10, 20, 30, 40, 50])
    drain_iter = v.drain(1, 4)
    val1 = next(drain_iter)  # Get first value (20)
    val2 = next(drain_iter)  # Get second value (30)
    expected_val1 = 20
    expected_val2 = 30
    assert val1 == expected_val1
    assert val2 == expected_val2
    # Don't consume the last element (40), let GC clean it up
    del drain_iter
    assert list(v) == [10, 50]


def test_drain_empty_immediately_gc() -> None:
    """Test drain with immediate garbage collection without consuming."""
    v = pc.Vec([1, 2, 3, 4])
    drain_iter = v.drain(1, 3)
    # Immediately delete without consuming anything
    del drain_iter
    assert list(v) == [1, 4]


def test_drain_full_consumption() -> None:
    """Test drain with full consumption of iterator."""
    v = pc.Vec([5, 6, 7])
    drained = v.drain(1, 2).collect()
    assert list(drained) == [6]
    assert list(v) == [5, 7]


def test_drain_entire_vector_gc() -> None:
    """Test draining entire vector via garbage collection."""
    v = pc.Vec([7, 8, 9])
    drain_iter = v.drain()
    next(drain_iter)  # consume only first
    del drain_iter
    assert list(v) == []
