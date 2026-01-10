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
