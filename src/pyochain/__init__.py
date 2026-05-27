"""pyochain - A functional programming library for Python."""

from ._dict import Dict
from ._iter import Iter, Position
from ._range import Range
from ._seq import Seq
from ._set import Set, SetMut
from ._sliceview import SliceView
from ._vec import Vec
from ._views import PyoItemsView, PyoKeysView, PyoValuesView
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
    "Position",
    "PyoItemsView",
    "PyoKeysView",
    "PyoValuesView",
    "Range",
    "Result",
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
