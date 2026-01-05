"""Type approaches experiments for Option and Result."""

from ._option import NONE, NoneOption, Option, Some
from ._result import Err, Ok, Result, ResultUnwrapError

__all__ = [
    "NONE",
    "Err",
    "NoneOption",
    "Ok",
    "Option",
    "Result",
    "ResultUnwrapError",
    "Some",
]
