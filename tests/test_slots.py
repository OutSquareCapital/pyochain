"""Tests for slot usage in pyochain classes."""

from collections.abc import Collection, Container, Iterable, Iterator, Reversible, Sized

import pyochain as pc
from pyochain import abc as pyoabc


def test_slots() -> None:
    assert _check_slots(pc.Iter(()))
    assert _check_slots(pc.Seq(()))
    assert _check_slots(pc.Set(()))
    assert _check_slots(pc.SetMut(()))
    assert _check_slots(pc.Dict[str, str](()))
    assert _check_slots(pc.Some(42))
    assert _check_slots(pc.Range(0, 1))
    assert _check_slots(pc.Vec(()))
    assert _check_slots(pc.NONE)
    assert _check_slots(pc.Err[int, object](42))
    assert _check_slots(pc.Ok[int, object](42))


def _check_slots(obj: object) -> bool:
    try:
        _x = obj.__dict__
        return False  # noqa: TRY300
    except AttributeError:
        return True


def test_abcs() -> None:

    assert issubclass(pyoabc.PyoIterable, Iterable)
    assert issubclass(pyoabc.PyoIterator, (pyoabc.PyoIterable, Iterable, Iterator))
    assert issubclass(pyoabc.PyoContainer, Container)
    assert issubclass(pyoabc.PyoSized, Sized)
    assert issubclass(
        pyoabc.PyoCollection,
        (pyoabc.PyoIterable, pyoabc.PyoContainer, pyoabc.PyoSized, Collection),
    )
    assert issubclass(pyoabc.PyoReversible, Reversible)


def test_inerhitance() -> None:
    assert issubclass(pc.Vec, (Iterable, pyoabc.PyoIterable))
    assert issubclass(pc.Iter, (Iterator, pyoabc.PyoIterator))
