from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Never, TypeIs


class OptionUnwrapError(RuntimeError): ...


class Option[T](ABC):
    @abstractmethod
    def is_some(self) -> TypeIs[Some[T]]:  # type: ignore[misc]
        """
        Returns `True` if the option is a `Some` value.

        Returns:
            `True` if the option is a `Some` variant, `False` otherwise.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE, Option
            >>> x: Option[int] = Some(2)
            >>> x.is_some()
            True
            >>> y: Option[int] = NONE
            >>> y.is_some()
            False
            ```
        """
        ...

    @abstractmethod
    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """
        Returns `True` if the option is a `None` value.

        Returns:
            `True` if the option is a `_None` variant, `False` otherwise.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE, Option
            >>> x: Option[int] = Some(2)
            >>> x.is_none()
            False
            >>> y: Option[int] = NONE
            >>> y.is_none()
            True
            ```
        """
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """
        Returns the contained `Some` value.

        Returns:
            The contained `Some` value.

        Raises:
            OptionUnwrapError: If the option is `None`.

        Example:
            ```python
            >>> from pyochain._results import Some
            >>> Some("car").unwrap()
            'car'
            ```
            ```python
            >>> from pyochain._results import NONE
            >>> NONE.unwrap()
            Traceback (most recent call last):
                ...
            pyochain._results._option.OptionUnwrapError: called `unwrap` on a `None`

            ```
        """
        ...

    def expect(self, msg: str) -> T:
        """
        Returns the contained `Some` value.
        Raises an exception with a provided message if the value is `None`.

        Args:
            msg: The message to include in the exception if the result is `None`.

        Returns:
            The contained `Some` value.

        Raises:
            OptionUnwrapError: If the result is `None`.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE
            >>> Some("value").expect("fruits are healthy")
            'value'
            >>> NONE.expect("fruits are healthy")
            Traceback (most recent call last):
                ...
            pyochain._results._option.OptionUnwrapError: fruits are healthy (called `expect` on a `None`)
            ```
        """
        if self.is_some():
            return self.unwrap()
        msg = f"{msg} (called `expect` on a `None`)"
        raise OptionUnwrapError(msg)

    def unwrap_or(self, default: T) -> T:
        """
        Returns the contained `Some` value or a provided default.

        Args:
            default: The value to return if the result is `None`.

        Returns:
            The contained `Some` value or the provided default.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE
            >>> Some("car").unwrap_or("bike")
            'car'
            >>> NONE.unwrap_or("bike")
            'bike'
            ```
        """
        return self.unwrap() if self.is_some() else default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """
        Returns the contained `Some` value or computes it from a function.

        Args:
            f: A function that returns a default value if the result is `None`.

        Returns:
            The contained `Some` value or the result of the function.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE
            >>> k = 10
            >>> Some(4).unwrap_or_else(lambda: 2 * k)
            4
            >>> NONE.unwrap_or_else(lambda: 2 * k)
            20
            ```
        """
        return self.unwrap() if self.is_some() else f()

    def map[U](self, f: Callable[[T], U]) -> Option[U]:
        """
        Maps an `Option[T]` to `Option[U]` by applying a function to a contained `Some` value,
        leaving a `None` value untouched.

        Args:
            f: The function to apply to the `Some` value.

        Returns:
            A new `Option` with the mapped value if `Some`, otherwise `None`.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE
            >>> Some("Hello, World!").map(len)
            Some(value=13)
            >>> NONE.map(len)
            NONE
            ```
        """
        if self.is_some():
            return Some(f(self.unwrap()))
        return NONE

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """
        Calls a function if the option is `Some`, otherwise returns `None`.
        Some languages call this operation flatmap.

        Args:
            f: The function to call with the `Some` value.

        Returns:
            The result of the function if `Some`, otherwise `None`.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE, Option
            >>> def sq(x: int) -> Option[int]:
            ...     return Some(x * x)
            >>> def nope(x: int) -> Option[int]:
            ...     return NONE
            >>> Some(2).and_then(sq).and_then(sq)
            Some(value=16)
            >>> Some(2).and_then(sq).and_then(nope)
            NONE
            >>> Some(2).and_then(nope).and_then(sq)
            NONE
            >>> NONE.and_then(sq).and_then(sq)
            NONE
            ```
        """
        if self.is_some():
            return f(self.unwrap())
        return NONE

    def or_else(self, f: Callable[[], Option[T]]) -> Option[T]:
        """
        Returns the option if it contains a value, otherwise calls a function and returns the result.

        Args:
            f: The function to call if the option is `None`.

        Returns:
            The original `Option` if it is `Some`, otherwise the result of the function.

        Example:
            ```python
            >>> from pyochain._results import Some, NONE, Option
            >>> def nobody() -> Option[str]:
            ...     return NONE
            >>> def vikings() -> Option[str]:
            ...     return Some("vikings")
            >>> Some("barbarians").or_else(vikings)
            Some(value='barbarians')
            >>> NONE.or_else(vikings)
            Some(value='vikings')
            >>> NONE.or_else(nobody)
            NONE
            ```
        """
        return self if self.is_some() else f()


@dataclass(slots=True)
class Some[T](Option[T]):
    value: T

    def is_some(self) -> TypeIs[Some[T]]:  # type: ignore[misc]
        """Returns `True` for `Some`."""
        return True

    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """Returns `False` for `Some`."""
        return False

    def unwrap(self) -> T:
        """
        Returns the contained value.

        Returns:
            The contained value.
        """
        return self.value


@dataclass(slots=True)
class _None(Option[Any]):
    def is_some(self) -> TypeIs[Some[Any]]:  # type: ignore[misc]
        """Returns `False` for `None`."""
        return False

    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """Returns `True` for `None`."""
        return True

    def unwrap(self) -> Never:
        """
        Raises `OptionUnwrapError` because there is no value.

        Raises:
            OptionUnwrapError: Always, since `None` contains no value.
        """
        raise OptionUnwrapError("called `unwrap` on a `None`")


NONE: Option[Any] = _None()
