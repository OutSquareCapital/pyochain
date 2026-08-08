"""This module is the actual "live" compiled module.

However, we mirror the Rust code layout in the stubs with multiple "false" private `.pyi` files.

Those are then re-exported in this module.

That way, we can :

- import from rs in the public __init__
- Support `from pyochain import foo` at runtime
- Handle partially migrated modules (python -> rust) without breaking the public API
"""

from collections.abc import Callable, Iterable
from types import NotImplementedType
from typing import Any, Self, overload, type_check_only

from _typeshed import SupportsRichComparison

from pyochain.abc import PyoIterator

from ._collections import Deque, Heap, HeapMax, HeapMin, PyoCounter, StableSet
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

@type_check_only
class InnerSorted[T, U]:
    len: int
    load: int

    def __new__(cls) -> Self: ...
    def clear(self) -> None: ...
    def contains(self, value: object) -> bool: ...
    def collapse_lists(self) -> Vec[Any]: ...
    def build_index(self) -> None: ...
    def add(self, value: T) -> None: ...
    def discard(self, value: T) -> None: ...
    def remove(self, value: T) -> None: ...
    def bisect_left(self, value: T) -> int: ...
    def bisect_right(self, value: T) -> int: ...
    def count(self, value: T) -> int: ...
    def index(
        self, value: T, start: int | None = None, stop: int | None = None
    ) -> int: ...
    def pop(self, index: int = -1) -> T: ...
    @overload
    def getitem(self, index: int) -> T: ...
    @overload
    def getitem(self, index: slice) -> Vec[T]: ...
    def getitem(self, index: int | slice) -> T | Vec[T]: ...
    def delitem(self, index: int | slice) -> None: ...
    def reset(self, load: int) -> None: ...
    def update(self, iterable: Iterable[T]) -> None: ...
    def eq(self, other: object) -> NotImplementedType | bool: ...
    def ne(self, other: object) -> NotImplementedType | bool: ...
    def lt(self, other: object) -> NotImplementedType | bool: ...
    def gt(self, other: object) -> NotImplementedType | bool: ...
    def le(self, other: object) -> NotImplementedType | bool: ...
    def ge(self, other: object) -> NotImplementedType | bool: ...
    def islice(
        self,
        start: int | None = None,
        stop: int | None = None,
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
    def iter(self) -> PyoIterator[T]: ...
    def reversed(self) -> PyoIterator[T]: ...
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...

class InnerLists[T, U](InnerSorted[T, U]):
    def __new__(cls) -> Self: ...

class InnerKeyLists[T, U, OT: SupportsRichComparison](InnerSorted[T, U]):
    key: KeyFunc[T, OT]
    def __new__(cls, key: KeyFunc[T, OT]) -> Self: ...
    def irange_key(
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        *,
        reverse: bool = False,
    ) -> PyoIterator[T]: ...
    def bisect_key_left(self, key: OT) -> int: ...
    def bisect_key_right(self, key: OT) -> int: ...

def check_sorted_list(data: InnerSorted[Any, Any]) -> None: ...
def check_sorted_key_list(data: InnerKeyLists[Any, Any, Any]) -> None: ...
def assert_sorted_list_empty(lst: InnerSorted[Any, Any]) -> None: ...

__all__ = [
    "NONE",
    "Deque",
    "Dict",
    "Err",
    "Heap",
    "HeapMax",
    "HeapMin",
    "InnerKeyLists",
    "InnerLists",
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
    "StableSet",
    "Vec",
    "option",
    "then_if_some",
    "then_if_true",
]
