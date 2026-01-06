"""pyochain - A functional programming library for Python."""

from . import traits
from ._dict import Dict
from ._iter import Iter, Seq, Set, SetMut, Unzipped, Vec
from ._option import NONE, NoneOption, Option, Some
from ._result import Err, Ok, Result, ResultUnwrapError

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
