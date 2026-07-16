"""pyochain - A functional programming library for Python."""

# NOTE: We need to keep `rs` import at first, otherwise it will crash.
# This import is what populate `sys.modules` to allow the other subsequents imports to work.
# TODO: Once the code is in pure rust we can proceed to delete the `sys` handling and probably simplify our import machinery

from pyochain.rs import (
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

from ._dict import Dict  # pyright: ignore[reportMissingModuleSource]
from ._range import Range  # pyright: ignore[reportMissingModuleSource]
from ._seq import Seq  # pyright: ignore[reportMissingModuleSource]
from ._set import Set, SetMut  # pyright: ignore[reportMissingModuleSource]
from ._sliceview import SliceView
from ._tools import Iter, Peekable  # pyright: ignore[reportMissingModuleSource]
from ._vec import Vec  # pyright: ignore[reportMissingModuleSource]
from .abc import PyoItemsView, PyoKeysView, PyoValuesView

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
