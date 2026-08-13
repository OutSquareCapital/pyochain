"""Re-export to allow to make it work with both package layouts, LSPs, etc...

Duplication of import here and in `__init__.py` is unfortunately necessary.
"""

from .core._dict import Dict
from .core._iterators import Iter, Peekable
from .core._option import (
    NONE,
    Null,
    Option,
    OptionType,
    OptionUnwrapError,
    Some,
    option,
)
from .core._range import Range
from .core._result import (
    Err,
    Ok,
    Result,
    ResultType,
    ResultUnwrapError,
    then_if_some,
    then_if_true,
)
from .core._seq import Seq
from .core._set import Set, SetMut
from .core._sliceview import SliceView
from .core._vec import Vec

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
