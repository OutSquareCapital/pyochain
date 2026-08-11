"""pyochain - A functional programming library for Python."""

from ._dict import Dict
from ._iterators import Iter, Peekable
from ._option import NONE, Null, Option, OptionType, OptionUnwrapError, Some, option
from ._range import Range
from ._result import (
    Err,
    Ok,
    Result,
    ResultType,
    ResultUnwrapError,
    then_if_some,
    then_if_true,
)
from ._seq import Seq
from ._set import Set, SetMut
from ._sliceview import SliceView
from ._vec import Vec

__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "Null",
    "Ok",
    "Option",
    "OptionType",
    "OptionUnwrapError",
    "Peekable",
    "Range",
    "Result",
    "ResultType",
    "ResultUnwrapError",
    "Seq",
    "Set",
    "SetMut",
    "SliceView",
    "Some",
    "Vec",
    "option",
    "then_if_some",
    "then_if_true",
]
