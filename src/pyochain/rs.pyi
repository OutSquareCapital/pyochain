"""This module is the actual "live" compiled module.

However, we mirror the Rust code layout in the stubs with multiple "false" private `.pyi` files.

Those are then re-exported in this module.

That way, we can :

- import from rs in the public __init__
- Support `from pyochain import foo` at runtime
- Handle partially migrated modules (python -> rust) without breaking the public API
"""

from collections.abc import Callable
from typing import Any

from _typeshed import SupportsRichComparison

from pyochain._collections._sorted_list import BaseSortedList

from ._collections import (
    Deque,
    Heap,
    HeapMax,
    HeapMin,
    PyoCounter,
    SortedKeyList,
    SortedList,
    StableSet,
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

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]
type SliceBounds = tuple[int, int, int, int]

def check_sorted_list(data: BaseSortedList[Any]) -> None: ...
def check_sorted_key_list(data: SortedKeyList[Any, Any]) -> None: ...
def assert_sorted_list_empty(lst: BaseSortedList[Any]) -> None: ...

__all__ = [
    "NONE",
    "Deque",
    "Dict",
    "Err",
    "Heap",
    "HeapMax",
    "HeapMin",
    "Iter",
    "Null",
    "Ok",
    "Option",
    "OptionType",
    "OptionUnwrapError",
    "Peekable",
    "PyoCounter",
    "Range",
    "Result",
    "ResultType",
    "ResultUnwrapError",
    "Seq",
    "Set",
    "SetMut",
    "SliceView",
    "Some",
    "SortedKeyList",
    "SortedList",
    "StableSet",
    "Vec",
    "option",
    "then_if_some",
    "then_if_true",
]
