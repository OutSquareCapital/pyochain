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
    option,
    then_if_some,
    then_if_true,
)

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
    "option",
    "then_if_some",
    "then_if_true",
    "traits",
]
