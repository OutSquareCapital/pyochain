"""pyochain - A functional programming library for Python."""

from . import _option as old_option
from . import _result as old_result
from . import traits
from ._dict import Dict
from ._iter import Iter, Peekable, Seq, Set, SetMut, Unzipped, Vec
from .rs import (
    NONE,
    Err,
    NoneOption,
    Ok,
    Option,
    OptionUnwrapError,
    Result,
    ResultUnwrapError,
    Some,
)

__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "NoneOption",
    "Ok",
    "Option",
    "OptionUnwrapError",
    "Peekable",
    "Result",
    "ResultUnwrapError",
    "Seq",
    "Set",
    "SetMut",
    "Some",
    "Unzipped",
    "Vec",
    "old_option",
    "old_result",
    "traits",
]
