import re
from collections.abc import (
    Collection,
    Container,
    Iterable,
    Iterator,
    Mapping,
    MappingView,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Reversible,
    Sequence,
    Sized,
)
from collections.abc import Set as AbstractSet
from functools import partial

import pytest

from pyochain import Seq, abc, core

check_other = partial(pytest.mark.parametrize, "other")

PYOITERATOR_PARENTS = Seq(Iterable, abc.PyoIterable, Iterator)
COLLECTION_PARENTS = Seq(
    abc.PyoIterable,
    abc.PyoContainer,
    abc.PyoSized,
    Collection,
    Container,
    Sized,
)
SET_PARENTS = COLLECTION_PARENTS.concat((abc.PyoSet, Collection, AbstractSet))
SEQUENCE_PARENTS = COLLECTION_PARENTS.concat((abc.PyoReversible, Reversible, Sequence))
MUTABLE_SET_PARENTS = SET_PARENTS.concat((abc.PyoMutableSet, MutableSet))
MUTABLE_SEQUENCE_PARENTS = SEQUENCE_PARENTS.concat((
    abc.PyoSequence,
    abc.PyoMutableSequence,
    MutableSequence,
))
MAPPING_PARENTS = (*COLLECTION_PARENTS, Mapping)
MUTABLE_MAPPING_PARENTS = (
    *MAPPING_PARENTS,
    abc.PyoMutableMapping,
    abc.PyoMapping,
    MutableMapping,
)

FAILING_PARENTS = core.Set[type](abc.PyoSized, abc.PyoContainer, abc.PyoReversible)

CURRENTLY_FAILING = re.compile(
    rf"({FAILING_PARENTS.iter().map(lambda x: x.__name__).join('|')})"
)
"""`PyoSized` and `PyoContainer` are currently failing due to pyo3 limitations for multiple inheritance."""
IGNORE_RAISE = pytest.raises(AssertionError, match=CURRENTLY_FAILING)


@check_other(PYOITERATOR_PARENTS)
def test_pyoiterator(other: type) -> None:
    assert issubclass(abc.PyoIterator, other)


@check_other(COLLECTION_PARENTS)
def test_collection(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(abc.PyoCollection, other)
        case _:
            assert issubclass(abc.PyoCollection, other)


@check_other(SEQUENCE_PARENTS)
def test_sequence(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(abc.PyoSequence, other)
        case _:
            assert issubclass(abc.PyoSequence, other)


@check_other(MUTABLE_SEQUENCE_PARENTS)
def test_mutable_sequence(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(abc.PyoMutableSequence, other)
        case _:
            assert issubclass(abc.PyoMutableSequence, other)


@check_other(SET_PARENTS)
def test_set(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(core.Set, other)
        case _:
            assert issubclass(core.Set, other)


@check_other(MUTABLE_SET_PARENTS)
def test_setmut(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(core.SetMut, other)
        case _:
            assert issubclass(core.SetMut, other)


@check_other((abc.PyoIterator, *PYOITERATOR_PARENTS))
def test_iter(other: type) -> None:
    assert issubclass(core.Iter, other)


@pytest.mark.parametrize("slf", (core.Seq, core.Range, core.SliceView))
@check_other([abc.PyoSequence, *SEQUENCE_PARENTS])
def test_concrete_sequences(slf: type, other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(slf, other)
        case _:
            assert issubclass(slf, other)


@check_other((abc.PyoMutableSequence, *MUTABLE_SEQUENCE_PARENTS))
def test_vec(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(core.Vec, other)
        case _:
            assert issubclass(core.Vec, other)


@check_other(MUTABLE_MAPPING_PARENTS)
def test_dict(other: type) -> None:
    match other:
        case _ if other in FAILING_PARENTS:
            with IGNORE_RAISE:
                assert issubclass(core.Dict, other)
        case _:
            assert issubclass(core.Dict, other)


@pytest.mark.parametrize(
    "classes",
    (
        (abc.PyoIterable, Iterable),
        (abc.PyoContainer, Container),
        (abc.PyoSized, Sized),
        (abc.PyoReversible, Reversible),
        (abc.PyoMappingView, MappingView),
        (abc.PyoMappingView, Sized),
    ),
)
def test_simple_abcs(classes: tuple[type, type]) -> None:
    assert issubclass(classes[0], classes[1])
