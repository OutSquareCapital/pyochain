"""Tests for SliceView."""

import pytest

from pyochain import SliceView


def test_basic() -> None:
    sv = SliceView([1, 2, 3])
    assert list(sv) == [1, 2, 3]


def test_with_start_stop() -> None:
    sv = SliceView([0, 1, 2, 3, 4], 1, 4)
    assert list(sv) == [1, 2, 3]


def test_with_step() -> None:
    sv = SliceView([0, 1, 2, 3, 4], 0, 5, 2)
    assert list(sv) == [0, 2, 4]


def test_with_slice_object() -> None:
    sv = SliceView([0, 1, 2, 3, 4], slice(1, 4))
    assert list(sv) == [1, 2, 3]


def test_negative_step() -> None:
    sv = SliceView([0, 1, 2, 3, 4], 4, None, -1)
    assert list(sv) == [4, 3, 2, 1, 0]


def test_string_base() -> None:
    sv = SliceView("hello")[1:4]
    assert list(sv) == ["e", "l", "l"]


def test_tuple_base() -> None:
    sv = SliceView((10, 20, 30, 40))[::2]
    assert list(sv) == [10, 30]


def test_positive_index() -> None:
    sv = SliceView([10, 20, 30])[:]
    assert sv[0] == 10
    assert sv[2] == 30


def test_negative_index() -> None:
    sv = SliceView([10, 20, 30])
    assert sv[-1] == 30
    assert sv[-3] == 10


def test_out_of_range() -> None:
    sv = SliceView([1, 2, 3])
    with pytest.raises(IndexError):
        _ = sv[10]


def test_index_into_strided_view() -> None:
    sv = SliceView(list(range(10)))[::3]
    assert sv[0] == 0
    assert sv[1] == 3
    assert sv[2] == 6


def test_slice_returns_self() -> None:
    sv = SliceView([1, 2, 3, 4, 5])
    assert isinstance(sv[1:3], SliceView)


def test_composed_slice() -> None:
    data = list(range(20))
    sv = SliceView(data)[2:][::3][1:4]
    assert list(sv) == [5, 8, 11]


def test_no_copy_on_slice() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)
    sv2 = sv[1:4]
    assert sv2.inner is data


def test_full_slice_is_same_base() -> None:
    data = [1, 2, 3]
    sv = SliceView(data)
    assert sv[:].inner is data


def test_negative_step_composition() -> None:
    data = list(range(10))
    sv = SliceView(data)[::-1][::2]
    assert list(sv) == [9, 7, 5, 3, 1]


def test_empty_slice() -> None:
    sv = SliceView([1, 2, 3])[5:10]
    assert list(sv) == []
    assert len(sv) == 0


def test_full() -> None:
    assert len(SliceView([1, 2, 3])) == 3


def test_partial() -> None:
    assert len(SliceView([1, 2, 3, 4, 5])[1:4]) == 3


def test_step() -> None:
    assert len(SliceView(list(range(10)))[::3]) == 4


def test_empty() -> None:
    assert len(SliceView(())) == 0


def test_reflects_base_mutation() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)
    assert len(sv) == 5
    data.append(6)
    assert len(sv) == 6


def test_setitem_int() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)[1:4]
    sv[0] = 99
    assert data == [1, 99, 3, 4, 5]


def test_setitem_slice() -> None:
    data = list(range(5))
    sv = SliceView(data)
    sv[1:4] = [10, 20, 30]
    assert data == [0, 10, 20, 30, 4]


def test_setitem_strided() -> None:
    data = list(range(10))
    sv = SliceView(data)[::2]
    sv[0] = 99
    assert data[0] == 99


def test_setitem_extended_slice_wrong_size() -> None:
    data = list(range(10))
    sv = SliceView(data)[::2]
    with pytest.raises(ValueError):  # noqa: PT011
        sv[0:3] = [1, 2]  # 3 slots, 2 values


def test_immutable_base_raises() -> None:
    sv = SliceView(range(5))
    with pytest.raises(TypeError):
        sv[0] = 99


def test_advance_basic() -> None:
    data = list(range(10))
    sv = SliceView(data, 0, 3)
    assert list(sv) == [0, 1, 2]
    _ = sv.advance(3)
    assert list(sv) == [3, 4, 5]


def test_advance_returns_self() -> None:
    sv = SliceView(list(range(10)), 0, 5)
    assert sv.advance(5) is sv


def test_advance_past_end_clamps() -> None:
    data = list(range(5))
    sv = SliceView(data, 0, 3)
    _ = sv.advance(100)
    assert list(sv) == []


def test_advance_negative() -> None:
    data = list(range(10))
    sv = SliceView(data, 5, 8)
    _ = sv.advance(-3)
    assert list(sv) == [2, 3, 4]


def test_sliding_window() -> None:
    data = list(range(12))
    sv = SliceView(data, 0, 4)
    result: list[list[int]] = []
    for _ in range(3):
        result.append(list(sv))
        _ = sv.advance(4)
    assert result == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]


def test_iter() -> None:
    sv = SliceView([10, 20, 30, 40])[1:3]
    assert list(sv) == [20, 30]


def test_contains() -> None:
    sv = SliceView([1, 2, 3, 4, 5])[1:4]
    assert 2 in sv
    assert 5 not in sv


def test_reversed() -> None:
    sv = SliceView([1, 2, 3, 4, 5])
    assert list(reversed(sv)) == [5, 4, 3, 2, 1]


def test_equal_to_list() -> None:
    sv = SliceView([1, 2, 3])
    assert sv == [1, 2, 3]


def test_equal_to_self() -> None:
    data = [1, 2, 3, 4]
    sv1 = SliceView(data)[1:3]
    sv2 = SliceView(data)[1:3]
    assert sv1 == sv2


def test_not_equal() -> None:
    sv = SliceView([1, 2, 3])
    assert sv != [1, 2, 4]


def test_unhashable() -> None:
    sv = SliceView([1, 2, 3])
    with pytest.raises(TypeError):
        _ = hash(sv)


def test_repr_contains_slice() -> None:
    sv = SliceView([1, 2, 3, 4])[1:3]
    r = repr(sv)
    assert "SliceView" in r
    assert "1:3" in r


def test_repr_full() -> None:
    sv = SliceView([1, 2, 3])
    r = repr(sv)
    assert "SliceView" in r


def test_tolist() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)[1:4]
    result = sv.iter().collect(list)
    assert result == [2, 3, 4]
    assert isinstance(result, list)
    result[0] = 99
    assert data[1] == 2  # original unchanged


def test_base_is_original() -> None:
    data = [1, 2, 3]
    sv = SliceView(data)
    assert sv.inner is data


def test_composed_base_is_original() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)[1:][::2]
    assert sv.inner is data


def test_write_through() -> None:
    data = [1, 2, 3, 4, 5]
    sv = SliceView(data)
    data[2] = 99
    assert sv[2] == 99


def test_append_reflected_in_len() -> None:
    data = [1, 2, 3]
    sv = SliceView(data)
    data.append(4)
    assert len(sv) == 4
