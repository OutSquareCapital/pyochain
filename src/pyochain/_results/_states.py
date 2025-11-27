from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Never

from ._option import Option, OptionUnwrapError
from ._result import Result, ResultUnwrapError


@dataclass(slots=True)
class Ok[T, E](Result[T, E]):
    """Represents a successful value.

    Attributes:
        value (T): The contained successful value.
    """

    value: T

    def is_ok(self) -> bool:
        return True

    def is_err(self) -> bool:
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

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
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

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(slots=True)
class NoneOption(Option[Any]):
    """Option variant representing the absence of a value."""

    def __repr__(self) -> str:
        return "NONE"

    def is_some(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def unwrap(self) -> Never:
        msg = "called `unwrap` on a `None`"
        raise OptionUnwrapError(msg)


NONE: Option[Any] = NoneOption()
"""Singleton instance representing the absence of a value."""
