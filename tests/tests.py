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


def test_retain_basic() -> None:
    """Test basic retain functionality."""
    v = pc.Vec([1, 2, 3, 4, 5])
    v.retain(lambda x: x % 2 == 0)
    assert list(v) == [2, 4]


def test_retain_preserves_order() -> None:
    """Test that retain preserves element order."""
    v = pc.Vec([5, 4, 3, 2, 1])
    v.retain(lambda x: x > 2)
    assert list(v) == [5, 4, 3]


def test_retain_empty_result() -> None:
    """Test retain that removes all elements."""
    v = pc.Vec([1, 2, 3, 4])
    v.retain(lambda x: x > 10)
    assert list(v) == []


def test_retain_all_kept() -> None:
    """Test retain where all elements are kept."""
    v = pc.Vec([1, 2, 3, 4])
    v.retain(lambda x: x > 0)
    assert list(v) == [1, 2, 3, 4]


def test_retain_with_stateful_predicate() -> None:
    """Test retain with external state-based predicate."""
    v = pc.Vec([1, 2, 3, 4, 5])
    keep = pc.Seq([False, True, True, False, True]).iter()
    v.retain(lambda _: next(keep))
    assert list(v) == [2, 3, 5]


def test_retain_no_intermediate_copy() -> None:
    """Verify that retain modifies the same Vec instance in place."""
    v = pc.Vec([1, 2, 3, 4])
    v_id_before = id(v._inner)  # type: ignore[private-access]
    v.retain(lambda x: x % 2 == 0)
    v_id_after = id(v._inner)  # type: ignore[private-access]
    # The underlying list should be the same object (in-place modification)
    assert v_id_before == v_id_after
    assert list(v) == [2, 4]


def test_truncate_basic() -> None:
    """Test basic truncate functionality."""
    v = pc.Vec([1, 2, 3, 4, 5])
    v.truncate(2)
    assert list(v) == [1, 2]


def test_truncate_to_zero() -> None:
    """Test truncate to zero length."""
    v = pc.Vec([1, 2, 3])
    v.truncate(0)
    assert list(v) == []


def test_truncate_no_effect() -> None:
    """Test truncate with length greater than current size."""
    v = pc.Vec([1, 2, 3])
    v.truncate(10)
    assert list(v) == [1, 2, 3]


def test_truncate_same_length() -> None:
    """Test truncate with length equal to current size."""
    v = pc.Vec([1, 2, 3])
    v.truncate(3)
    assert list(v) == [1, 2, 3]


def test_truncate_no_intermediate_copy() -> None:
    """Verify that truncate modifies the same Vec instance in place."""
    v = pc.Vec([1, 2, 3, 4, 5])
    v_id_before = id(v._inner)  # type: ignore[private-access]
    v.truncate(2)
    v_id_after = id(v._inner)  # type: ignore[private-access]
    # The underlying list should be the same object (in-place modification)
    assert v_id_before == v_id_after
    assert list(v) == [1, 2]


def test_extract_if_basic() -> None:
    """Test basic extract_if functionality."""
    v = pc.Vec([1, 2, 3, 4, 5])
    extracted = v.extract_if(lambda x: x % 2 == 0).collect()
    # After extracting, only odd elements remain
    assert list(v) == [1, 3, 5]
    # Extracted elements should be even
    assert list(extracted) == [2, 4]


def test_extract_if_with_range() -> None:
    """Test extract_if with specified range."""
    v = pc.Vec([1, 2, 3, 4, 5])
    extracted = v.extract_if(lambda x: x % 2 == 0, start=1, end=4).collect()
    # Only indices 1-3 are affected, so element 2 and 4 are extracted
    assert list(v) == [1, 3, 5]
    assert list(extracted) == [2, 4]


def test_extract_if_empty_result() -> None:
    """Test extract_if that matches no elements."""
    v = pc.Vec([1, 2, 3, 4])
    extracted = v.extract_if(lambda x: x > 10).collect()
    assert list(extracted) == []
    assert list(v) == [1, 2, 3, 4]


def test_extract_if_all_match() -> None:
    """Test extract_if that matches all elements."""
    v = pc.Vec([1, 2, 3, 4])
    extracted = v.extract_if(lambda x: x > 0).collect()
    assert list(extracted) == [1, 2, 3, 4]
    assert list(v) == []


def test_extract_if_partial_consumption() -> None:
    """Test extract_if with partial consumption."""
    v = pc.Vec([1, 2, 3, 4, 5])
    extract_iter = v.extract_if(lambda x: x % 2 == 0)
    # Consume only first extracted element (will be 2 since we collect first)
    first = next(extract_iter)
    assert first == 2
    # Consume rest
    remaining = list(extract_iter)
    assert remaining == [4]
    assert list(v) == [1, 3, 5]


def test_memory_efficiency_retain() -> None:
    """Test that retain doesn't create unnecessary allocations."""
    original = list(range(1000))
    v = pc.Vec(original)
    original_list_id = id(v._inner)  # type: ignore[private-access]

    # Retain only even numbers
    v.retain(lambda x: x % 2 == 0)

    # Verify in-place modification
    assert id(v._inner) == original_list_id  # type: ignore[private-access]
    assert len(v) == 500
    assert all(x % 2 == 0 for x in v)


def test_memory_efficiency_truncate() -> None:
    """Test that truncate doesn't create unnecessary allocations."""
    v = pc.Vec(list(range(100)))
    original_list_id = id(v._inner)  # type: ignore[private-access]

    v.truncate(50)

    # Verify in-place modification
    assert id(v._inner) == original_list_id  # type: ignore[private-access]
    assert len(v) == 50


def test_chained_memory_operations() -> None:
    """Test multiple in-place operations in sequence."""
    v = pc.Vec([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    original_id = id(v._inner)  # type: ignore[private-access]

    # First retain
    v.retain(lambda x: x > 2)
    assert id(v._inner) == original_id  # type: ignore[private-access]

    # Then truncate
    v.truncate(4)
    assert id(v._inner) == original_id  # type: ignore[private-access]

    # Verify final state
    assert list(v) == [3, 4, 5, 6]
