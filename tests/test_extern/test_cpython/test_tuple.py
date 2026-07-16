import pickle
import sys
from collections.abc import Iterator

import pytest

from pyochain import Seq

RUN_ALL_HASH_TESTS = False
JUST_SHOW_HASH_RESULTS = False  # if RUN_ALL_HASH_TESTS, just display
NHASHBITS = sys.hash_info.width


def test_getitem_error() -> None:
    t = Seq(())
    msg = "tuple indices must be integers or slices"
    with pytest.raises(TypeError, match=msg):
        t["a"]  # pyright: ignore[reportCallIssue, reportArgumentType]


# TODO: Make the behavior identical to tuple, which should make the two `seq is not seq_from_*` fail.
def test_constructors() -> None:
    """Contrary to `tuple`, the constructor won't return the same object if the argument is already a `Seq`.

    However, the `inner` attribute will be the same, so that the underlying data is not copied.
    """
    assert Seq[int](()) == ()
    seq = Seq((0, 1, 2, 3))
    seq_from_seq = Seq(seq)
    seq_from_tup = Seq(seq.inner)
    assert seq.inner is seq_from_seq.inner
    assert seq.inner is seq_from_tup.inner
    assert seq is not seq_from_seq
    assert seq is not seq_from_tup
    assert Seq([]) == ()
    assert Seq([0, 1, 2, 3]) == (0, 1, 2, 3)
    assert Seq("") == ()
    assert Seq("spam") == ("s", "p", "a", "m")
    assert Seq(x for x in range(10) if x % 2) == (1, 3, 5, 7, 9)


def test_keyword_args() -> None:
    with pytest.raises(TypeError, match="keyword argument"):
        _ = Seq(sequence=())  # pyright: ignore[reportCallIssue, reportUnknownVariableType]


def test_truth() -> None:
    assert not Seq(())
    assert Seq((42,))


def test_len() -> None:
    assert len(Seq(())) == 0
    assert len(Seq((0,))) == 1
    assert len(Seq((0, 1, 2))) == 3


def test_iadd() -> None:
    u = Seq((0, 1))
    u2 = u
    u += Seq((2, 3))
    assert u is not u2


def test_imul() -> None:
    u = Seq((0, 1))
    u2 = u
    u *= 3
    assert u is not u2


def test_seq_resize_bug() -> None:
    # Check that a specific bug in _PyTuple_Resize() is squashed.
    def f() -> Iterator[int]:
        yield from range(1000)

    assert list(Seq(f())) == list(range(1000))


# We expect Seqs whose base components have deterministic hashes to
# have deterministic hashes too - and, indeed, the same hashes across
# platforms with hash codes of the same bit width.
def test_hash_exact() -> None:
    def check_one_exact(t: Seq[object], e32: int, e64: int) -> None:
        got = hash(Seq(t))
        expected = e32 if NHASHBITS == 32 else e64
        if got != expected:
            msg = f"FAIL hash({Seq(t)!r}) == {got} != {expected}"
            raise AssertionError(msg)

    check_one_exact(Seq(()), 750394483, 5740354900026072187)
    check_one_exact(Seq((0,)), 1214856301, -8753497827991233192)
    check_one_exact(Seq((0, 0)), -168982784, -8458139203682520985)
    check_one_exact(Seq((0.5,)), 2077348973, -408149959306781352)
    check_one_exact(Seq((0.5, (), (-2, 3, (4, 6)))), 714642271, -1845940830829704396)


def test_repr() -> None:
    l0 = Seq[int](())
    l2 = Seq((0, 1, 2))
    a0 = Seq(l0)
    a2 = Seq(l2)

    assert str(a0) == repr(l0)
    assert str(a2) == repr(l2)
    assert repr(a0) == "Seq()"
    assert repr(a2) == "Seq(0, 1, 2)"


@pytest.mark.skip(reason="We don't handle recursive repr yet")
def test_repr_large() -> None:
    # Check the repr of large list objects
    def check(n: int) -> None:
        lst = Seq((0,) * n)
        s = repr(lst)
        assert s == "Seq(" + ", ".join(["0"] * n) + ")"

    check(10)  # check our checking code
    check(1000000)


@pytest.mark.skip("Pyo3 doesn't support pickling yet")
def test_iterator_pickle() -> None:
    # Userlist iterators don't support pickling yet since
    # they are based on generators.
    data = Seq([4, 5, 6, 7])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        itorg = iter(data)
        d = pickle.dumps(itorg, proto)
        it = pickle.loads(d)  # pyright: ignore[reportAny]
        assert type(itorg) is type(it)  # pyright: ignore[reportAny]
        assert Seq(it) == Seq(data)  # pyright: ignore[reportAny]

        it = pickle.loads(d)  # pyright: ignore[reportAny]
        next(it)  # pyright: ignore[reportAny]
        d = pickle.dumps(it, proto)
        assert Seq(it) == Seq(data)[1:]  # pyright: ignore[reportAny]


@pytest.mark.skip("Pyo3 doesn't support pickling yet")
def test_reversed_pickle() -> None:
    data = Seq([4, 5, 6, 7])
    for proto in range(pickle.HIGHEST_PROTOCOL + 1):
        itorg = reversed(data)
        d = pickle.dumps(itorg, proto)
        it = pickle.loads(d)  # pyright: ignore[reportAny]
        assert type(itorg) is type(it)  # pyright: ignore[reportAny]
        assert Seq(it) == Seq(reversed(data))  # pyright: ignore[reportAny]

        it = pickle.loads(d)  # pyright: ignore[reportAny]
        next(it)  # pyright: ignore[reportAny]
        d = pickle.dumps(it, proto)
        assert Seq(it) == Seq(reversed(data))[1:]  # pyright: ignore[reportAny]


def test_lexicographic_ordering() -> None:
    # Issue 21100
    a = Seq([1, 2])
    b = Seq([1, 2, 0])
    c = Seq([1, 3])
    assert a < b
    assert b < c
