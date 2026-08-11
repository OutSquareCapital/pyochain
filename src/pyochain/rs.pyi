"""This module is the actual "live" compiled module.

However, we mirror the Rust code layout in the stubs with multiple "false" private `.pyi` files.

Those are then re-exported in this module.

That way, we can :

- import from rs in the public __init__
- Support `from pyochain import foo` at runtime
- Handle partially migrated modules (python -> rust) without breaking the public API
"""

from typing import Any

from pyochain.collections._sorted import (
    BaseSortedDict,
    BaseSortedList,
    BaseSortedSet,
    SortedKeyList,
)

from ._dict import Dict
from ._iterators import Iter, Peekable
from ._option import NONE, Null, Option, OptionType, OptionUnwrapError, Some, option
from ._range import Range
from ._result import (
    Err,
    Ok,
    Result,
    ResultType,
    ResultUnwrapError,
    then_if_some,
    then_if_true,
)
from ._seq import Seq
from ._set import Set, SetMut
from ._sliceview import SliceView
from ._vec import Vec

def check_sorted_list(data: BaseSortedList[Any]) -> None: ...
def check_sorted_set(data: BaseSortedSet[Any]) -> None: ...
def check_sorted_key_list(data: SortedKeyList[Any, Any]) -> None: ...
def assert_sorted_list_empty(lst: BaseSortedList[Any]) -> None: ...
def check_sorted_dict(data: BaseSortedDict[Any, Any]) -> None: ...

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
