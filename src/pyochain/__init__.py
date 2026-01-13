"""pyochain - A functional programming library for Python."""

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
    "traits",
]
