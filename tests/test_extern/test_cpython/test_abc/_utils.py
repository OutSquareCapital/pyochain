from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Literal, override

import pytest

from pyochain.abc import PyoMutableSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


def fn(_s: object, *_args: object) -> int:
    return 0


def validate_abstract_methods(abc: type, *names: str) -> None:
    methodstubs = dict.fromkeys(names, fn)

    # everything should work will all required methods are present
    foo = type("C", (abc,), methodstubs)
    _ = foo()

    # instantiation should fail if a required method is missing
    for name in names:
        stubs = methodstubs.copy()
        del stubs[name]
        bar = type("C", (abc,), stubs)
        with pytest.raises(TypeError):
            _ = bar()


def validate_isinstance(abc: type, name: str) -> None:
    def stub(_s: object, *_args: object) -> int:
        return 0

    foo = type("C", (object,), {"__hash__": None})
    setattr(foo, name, stub)
    assert isinstance(foo(), abc)
    assert issubclass(foo, abc)

    foo = type("C", (object,), {"__hash__": None})
    assert not isinstance(foo(), abc)
    assert not issubclass(foo, abc)


type EqFn = Callable[[Any, object], bool]


def validate_comparison(instance: object) -> None:
    ops = ["lt", "gt", "le", "ge", "ne", "or", "and", "xor", "sub"]
    operators: dict[str, Callable[[object, object], object]] = {}
    for op in ops:
        name = "__" + op + "__"
        operators[name] = getattr(operator, name)

    class Other:  # ruff:ignore[eq-without-hash]
        def __init__(self) -> None:
            self.right_side: bool = False

        @override
        def __eq__(self, other: object) -> Literal[True]:
            self.right_side = True
            return True

        __lt__: EqFn = __eq__
        __gt__: EqFn = __eq__
        __le__: EqFn = __eq__
        __ge__: EqFn = __eq__
        __ne__: EqFn = __eq__
        __ror__: EqFn = __eq__
        __rand__: EqFn = __eq__
        __rxor__: EqFn = __eq__
        __rsub__: EqFn = __eq__

    for name, op in operators.items():
        if not hasattr(instance, name):
            continue
        other = Other()
        _ = op(instance, other)
        assert other.right_side, f"Right side not called for {type(instance)}.{name}"


class WithSet(PyoMutableSet[object]):
    def __init__(self, it: Iterable[object] = ()) -> None:
        self.data: set[object] = set(it)

    @override
    def __len__(self) -> int:
        return len(self.data)

    @override
    def __iter__(self) -> Iterator[object]:
        return iter(self.data)

    @override
    def __contains__(self, item: object) -> bool:
        return item in self.data

    @override
    def add(self, item: object) -> None:
        self.data.add(item)

    @override
    def discard(self, item: object) -> None:
        self.data.discard(item)


class _NeverEq:
    """
    Object that is not equal to anything.
    """

    @override
    def __eq__(self, other: object) -> Literal[False]:
        return False

    @override
    def __ne__(self, other: object) -> Literal[True]:
        return True

    @override
    def __hash__(self) -> Literal[1]:
        return 1


NEVER_EQ = _NeverEq()
