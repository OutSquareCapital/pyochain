from pyochain import Seq, Vec


def test_drain_partial_consumption_then_gc() -> None:
    v = Vec([1, 2, 3, 4, 5])
    drain_iter = v.drain(1, 4)
    assert next(drain_iter) == 2
    del drain_iter
    assert v == [1, 5]


def test_drain_no_args_partial_consumption_then_gc() -> None:
    v = Vec([1, 2, 3, 4, 5])
    drain_iter = v.drain()
    _ = next(drain_iter)
    del drain_iter
    assert v == []


def test_drain_no_consumption_gc() -> None:
    v = Vec([1, 2, 3, 4])
    drain_iter = v.drain(1, 3)
    del drain_iter
    assert v == [1, 4]


def test_drain_full_consumption() -> None:
    v = Vec([5, 6, 7])
    drained = v.drain(1, 2).collect(Seq)
    assert list(drained) == [6]
    assert v == [5, 7]


def test_retain_basic() -> None:
    v = Vec([1, 2, 3, 4, 5])
    v.retain(lambda x: x % 2 == 0)
    assert v == [2, 4]


def test_retain_empty_result() -> None:
    v = Vec([1, 2, 3, 4])
    v.retain(lambda x: x > 10)
    assert v == []


def test_retain_all_kept() -> None:
    v = Vec([1, 2, 3, 4])
    v.retain(lambda x: x > 0)
    assert v == [1, 2, 3, 4]


def test_retain_in_place() -> None:
    v = Vec([1, 2, 3, 4])
    inner_id = id(v.inner)
    v.retain(lambda x: x % 2 == 0)
    assert id(v.inner) == inner_id
    assert v == [2, 4]


def test_truncate_basic() -> None:
    v = Vec([1, 2, 3, 4, 5])
    v.truncate(2)
    assert v == [1, 2]


def test_truncate_to_zero() -> None:
    v = Vec([1, 2, 3])
    v.truncate(0)
    assert v == []


def test_truncate_no_effect() -> None:
    v = Vec([1, 2, 3])
    v.truncate(10)
    assert v == [1, 2, 3]


def test_truncate_same_length() -> None:
    v = Vec([1, 2, 3])
    v.truncate(3)
    assert v == [1, 2, 3]


def test_truncate_in_place() -> None:
    v = Vec([1, 2, 3, 4, 5])
    inner_id = id(v.inner)
    v.truncate(2)
    assert id(v.inner) == inner_id
    assert v == [1, 2]


def test_extract_if_basic() -> None:
    v = Vec([1, 2, 3, 4, 5])
    extracted = v.extract_if(lambda x: x % 2 == 0).collect(Seq)
    assert v == [1, 3, 5]
    assert list(extracted) == [2, 4]


def test_extract_if_with_range() -> None:
    v = Vec([1, 2, 3, 4, 5])
    extracted = v.extract_if(lambda x: x % 2 == 0, start=1, end=4).collect(Seq)
    assert v == [1, 3, 5]
    assert list(extracted) == [2, 4]


def test_extract_if_empty_result() -> None:
    v = Vec([1, 2, 3, 4])
    extracted = v.extract_if(lambda x: x > 10).collect(Seq)
    assert list(extracted) == []
    assert v == [1, 2, 3, 4]


def test_extract_if_all_match() -> None:
    v = Vec([1, 2, 3, 4])
    extracted = v.extract_if(lambda x: x > 0).collect(Seq)
    assert list(extracted) == [1, 2, 3, 4]
    assert v == []


def test_extract_if_partial_consumption() -> None:
    v = Vec([1, 2, 3, 4, 5])
    extract_iter = v.extract_if(lambda x: x % 2 == 0)
    assert next(extract_iter) == 2
    assert list(extract_iter) == [4]
    assert v == [1, 3, 5]
