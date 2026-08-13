"""Re-export to allow to make it work with both package layouts, LSPs, etc...

Duplication of import here and in `__init__.py` is unfortunately necessary.
"""

from . import abc, collections
from .core import (
    NONE,
    Dict,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    OptionType,
    OptionUnwrapError,
    Peekable,
    Range,
    Result,
    ResultType,
    ResultUnwrapError,
    Seq,
    Set,
    SetMut,
    SliceView,
    Some,
    Vec,
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
    "abc",
    "collections",
    "option",
    "then_if_some",
    "then_if_true",
]
