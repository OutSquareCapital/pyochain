"""pyochain - A functional programming library for Python."""

from . import traits
from ._dict import Dict
from ._iter import Iter, Peekable, Seq, Set, SetMut, Unzipped, Vec
from ._range import Range
from .rs import (
    NONE,
    Err,
    Null,
    Ok,
    Option,
    OptionUnwrapError,
    Result,
    ResultUnwrapError,
    Some,
)

abc = traits  # noqa: RUF067
__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "Null",
    "Ok",
    "Option",
    "OptionUnwrapError",
    "Peekable",
    "Range",
    "Result",
    "ResultUnwrapError",
    "Seq",
    "Set",
    "SetMut",
    "Some",
    "Unzipped",
    "Vec",
    "abc",
    "traits",
]
