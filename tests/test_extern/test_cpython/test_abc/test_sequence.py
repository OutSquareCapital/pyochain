from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

import pytest

from pyochain import Range, Seq, Set, Vec
from pyochain.abc import PyoSequence

from ._utils import validate_abstract_methods

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_sequence_ok() -> None:
    for sample in [Seq, Vec]:
        assert isinstance(sample(()), PyoSequence)
        assert issubclass(sample, PyoSequence)
    assert isinstance(Range(10), PyoSequence)
    assert issubclass(Range, PyoSequence)


@pytest.mark.skip(reason="No metaclass behavior in Pyo3 yet")
def test_sequence_abstract_methods() -> None:
    validate_abstract_methods(PyoSequence, "__len__", "__getitem__")


def test_sequence_not() -> None:
    """Note: Contrary to python collections.abc.Sequence, PyoSequence does not consider memoryview or str to be a sequence."""
    assert not isinstance(memoryview(b""), PyoSequence)
    assert not issubclass(memoryview, PyoSequence)
    assert not issubclass(str, PyoSequence)


class SequenceSubclass(PyoSequence[object]):
    def __init__(self, seq: Sequence[object] = ()) -> None:
        self.seq: Sequence[object] = seq

    @override
    def __getitem__(self, index: int) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.seq[index]

    @override
    def __len__(self) -> int:
        return len(self.seq)


def test_sequence_mixins() -> None:
    # Compare PyoSequence.index() behavior to (list|str).index() behavior

    native_seq = Seq("abracadabra")
    native_vec = Vec("abracadabra")
    natives = (native_seq, native_vec)
    for native in natives:
        letters = Set[str](native).union({"z"})
        seqseq = SequenceSubclass(native)

        for letter in letters:
            _assert_index_same(native, seqseq, (letter,))
            for start in range(-3, native.len() + 3):
                _assert_index_same(native, seqseq, (letter, start))
                for stop in range(-3, native.len() + 3):
                    _assert_index_same(native, seqseq, (letter, start, stop))


def _assert_index_same(
    seq1: PyoSequence[object],
    seq2: PyoSequence[object],
    index_args: tuple[Any, ...],
) -> None:
    try:
        expected = seq1.index(*index_args)  # pyright: ignore[reportAny]
    except ValueError:
        with pytest.raises(ValueError):
            _ = seq2.index(*index_args)  # pyright: ignore[reportAny]
    else:
        actual = seq2.index(*index_args)  # pyright: ignore[reportAny]
        assert actual == expected, f"{seq1!r}.index{index_args}"
