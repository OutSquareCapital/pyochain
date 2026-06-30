from collections.abc import Collection, Container, Iterable, Iterator, Reversible, Sized
from functools import partial

import pytest

import pyochain as pc
from pyochain.abc import (
    PyoCollection,
    PyoContainer,
    PyoIterable,
    PyoIterator,
    PyoReversible,
    PyoSized,
)


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


check_other = partial(pytest.mark.parametrize, "other")

PYOITERATOR_PARENTS = [Iterable, PyoIterable, Iterator]
COLLECTION_PARENTS = [PyoIterable, PyoContainer, PyoSized, Collection, Container, Sized]


@check_other(PYOITERATOR_PARENTS)
def test_pyoiterator(other: type) -> None:
    assert issubclass(PyoIterator, other)


@check_other(COLLECTION_PARENTS)
def test_collection(other: type) -> None:
    assert issubclass(PyoCollection, other)


@check_other([PyoIterator, *PYOITERATOR_PARENTS])
def test_iter(other: type) -> None:
    assert issubclass(pc.Iter, other)


@check_other([PyoCollection, *COLLECTION_PARENTS])
def test_vec(other: type) -> None:
    assert issubclass(pc.Vec, other)


@pytest.mark.parametrize(
    "classes",
    [
        (PyoIterable, Iterable),
        (PyoContainer, Container),
        (PyoSized, Sized),
        (PyoReversible, Reversible),
    ],
)
def test_simple_abcs(classes: tuple[type, type]) -> None:
    assert issubclass(classes[0], classes[1])
