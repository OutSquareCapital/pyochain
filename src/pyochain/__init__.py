"""pyochain - A functional programming library for Python."""

# NOTE: We need to keep `rs` import at first, otherwise it will crash.
# This import is what populate `sys.modules` to allow the other subsequents imports to work.
# TODO: Once the code is in pure rust we can proceed to delete the `sys` handling and probably simplify our import machinery
from pyochain.rs import (
    NONE,
    Dict,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    OptionUnwrapError,
    Peekable,
    Range,
    Result,
    ResultUnwrapError,
    Seq,
    Set,
    SetMut,
    Some,
    Vec,
    option,
    then_if_some,
    then_if_true,
)

from ._sliceview import SliceView

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
