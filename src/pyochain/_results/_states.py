from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Concatenate, Never, TypeIs

from .._core import CommonBase
from ._option import Option, OptionUnwrapError
from ._result import Result, ResultUnwrapError


@dataclass(slots=True)
class Ok[T, E](Result[T, E]):
    """Represents a successful value.

    Attributes:
        value (T): The contained successful value.
    """

    value: T

    def is_ok(self) -> TypeIs[Ok[T, E]]:  # type: ignore[misc]
        return True

    def is_err(self) -> TypeIs[Err[T, E]]:  # type: ignore[misc]
        return False

    def unwrap(self) -> T:
        return self.value

    def unwrap_err(self) -> Never:
        msg = "called `unwrap_err` on Ok"
        raise ResultUnwrapError(msg)


@dataclass(slots=True)
class Err[T, E](Result[T, E]):
    """Represents an error value.

    Attributes:
        error (E): The contained error value.
    """

    error: E

    def is_ok(self) -> TypeIs[Ok[T, E]]:  # type: ignore[misc]
        return False

    def is_err(self) -> TypeIs[Err[T, E]]:  # type: ignore[misc]
        return True

    def unwrap(self) -> Never:
        msg = f"called `unwrap` on Err: {self.error!r}"
        raise ResultUnwrapError(msg)

    def unwrap_err(self) -> E:
        return self.error


@dataclass(slots=True)
class Some[T](Option[T]):
    """Option variant representing the presence of a value.

    Attributes:
        value (T): The contained value.

    Example:
    ```python
    >>> import pyochain as pc
    >>> pc.Some(42)
    Some(value=42)

    ```

    """

    value: T

    def is_some(self) -> TypeIs[Some[T]]:  # type: ignore[misc]
        return True

    def is_none(self) -> TypeIs[NoneOption]:  # type: ignore[misc]
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(slots=True)
class NoneOption(Option[Any]):
    """Option variant representing the absence of a value."""

    def __repr__(self) -> str:
        return "NONE"

    def is_some(self) -> TypeIs[Some[Any]]:  # type: ignore[misc]
        return False

    def is_none(self) -> TypeIs[NoneOption]:  # type: ignore[misc]
        return True

    def unwrap(self) -> Never:
        msg = "called `unwrap` on a `None`"
        raise OptionUnwrapError(msg)


NONE: Option[Any] = NoneOption()
"""Singleton instance representing the absence of a value."""


class Wrapper[T](CommonBase[T]):
    """A generic Wrapper for any type.

    The pipe into method is implemented to return a Wrapper of the result type.

    This class is intended for use with other types/implementations that do not support the fluent/functional style.

    This allow the use of a consistent code style across the code base.

    """

    def apply[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Wrapper[R]:
        return Wrapper(self.into(func, *args, **kwargs))
