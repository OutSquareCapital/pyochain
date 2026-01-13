"""pyochain - A functional programming library for Python."""

from . import traits
from ._dict import Dict
from ._iter import Iter, Seq, Set, SetMut, Unzipped, Vec
from .rs import NONE, Err, NoneOption, Ok, Option, Result, ResultUnwrapError, Some

__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "NoneOption",
    "Ok",
    "Option",
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
