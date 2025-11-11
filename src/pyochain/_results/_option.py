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
        Returns True if the option is a Some value.

        This method is used to check if the Option contains a value (Some) or not (None).
        Equivalent to Rust's Option::is_some().
        """
        ...

    @abstractmethod
    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """
        Returns True if the option is a None value.

        This method is used to check if the Option does not contain a value.
        Equivalent to Rust's Option::is_none().
        """
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """
        Returns the contained value if the option is Some, otherwise raises OptionUnwrapError.

        This method is equivalent to Rust's Option::unwrap(). It should only be called when it is known that the Option is Some.
        """
        ...

    def expect(self, msg: str) -> T:
        """
        Returns the contained value if the option is Some, otherwise raises OptionUnwrapError with a custom message.

        Args:
            msg: The message to display if the Option is None.

        Returns:
            The contained value if Some.

        Raises:
            OptionUnwrapError: If the Option is None, with the provided message.

        Equivalent to Rust's Option::expect().
        """
        if self.is_some():
            return self.unwrap()
        msg = f"{msg} (called `expect` on a `None`)"
        raise OptionUnwrapError(msg)

    def unwrap_or(self, default: T) -> T:
        """
        Returns the contained value if Some, otherwise returns the provided default.

        Args:
            default: The value to return if the Option is None.

        Returns:
            The contained value or the default.

        Equivalent to Rust's Option::unwrap_or().
        """
        return self.unwrap() if self.is_some() else default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """
        Returns the contained value if Some, otherwise computes it from a closure.

        Args:
            f: A callable that returns a value if the Option is None.

        Returns:
            The contained value or the result of f().

        Equivalent to Rust's Option::unwrap_or_else().
        """
        return self.unwrap() if self.is_some() else f()

    def map[U](self, f: Callable[[T], U]) -> Option[U]:
        """
        Maps an Option[T] to Option[U] by applying a function to the contained value.

        Args:
            f: A callable that takes the contained value and returns a new value.

        Returns:
            Option[U]: Some(f(value)) if Some, otherwise None.

        Equivalent to Rust's Option::map().
        """
        if self.is_some():
            return Some(f(self.unwrap()))
        return NONE

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """
        Returns None if the option is None, otherwise calls f with the contained value and returns the result.

        Args:
            f: A callable that takes the contained value and returns an Option.

        Returns:
            Option[U]: The result of f(value) if Some, otherwise None.

        Equivalent to Rust's Option::and_then().
        """
        if self.is_some():
            return f(self.unwrap())
        return NONE

    def or_else(self, f: Callable[[], Option[T]]) -> Option[T]:
        """
        Returns the option if it contains a value, otherwise calls f and returns the result.

        Args:
            f: A callable that returns an Option.

        Returns:
            Option[T]: self if Some, otherwise the result of f().

        Equivalent to Rust's Option::or_else().
        """
        return self if self.is_some() else f()


@dataclass(slots=True)
class Some[T](Option[T]):
    value: T

    def is_some(self) -> TypeIs[Some[T]]:  # type: ignore[misc]
        """
        Always returns True for Some.

        Equivalent to Rust's Option::is_some().
        """
        return True

    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """
        Always returns False for Some.

        Equivalent to Rust's Option::is_none().
        """
        return False

    def unwrap(self) -> T:
        """
        Returns the contained value.

        Returns:
            The contained value.

        Equivalent to Rust's Option::unwrap() for Some.
        """
        return self.value


@dataclass(slots=True)
class _None(Option[Any]):
    def is_some(self) -> TypeIs[Some[Any]]:  # type: ignore[misc]
        """
        Always returns False for None.

        Equivalent to Rust's Option::is_some().
        """
        return False

    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """
        Always returns True for None.

        Equivalent to Rust's Option::is_none().
        """
        return True

    def unwrap(self) -> Never:
        """
        Raises OptionUnwrapError because there is no value.

        Raises:
            OptionUnwrapError: Always, since None contains no value.

        Equivalent to Rust's Option::unwrap() for None.
        """
        raise OptionUnwrapError("called `unwrap` on a `None`")


NONE: Option[Any] = _None()
