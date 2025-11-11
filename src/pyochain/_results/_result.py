from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Concatenate, Never, TypeIs, cast

from .._core import CommonBase
from ._option import NONE, Option, Some


class ResultUnwrapError(RuntimeError): ...


class Result[T, E](ABC):
    @abstractmethod
    def is_ok(self) -> TypeIs[Ok[T, E]]:  # type: ignore[misc]
        """
        Returns True if the result is Ok.

        Equivalent to Rust's Result::is_ok().
        """
        ...

    @abstractmethod
    def is_err(self) -> TypeIs[Err[T, E]]:  # type: ignore[misc]
        """
        Returns True if the result is Err.

        Equivalent to Rust's Result::is_err().
        """
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """
        Returns the contained Ok value, or raises ResultUnwrapError if the result is Err.

        Equivalent to Rust's Result::unwrap().
        """
        ...

    @abstractmethod
    def unwrap_err(self) -> E:
        """
        Returns the contained Err value, or raises ResultUnwrapError if the result is Ok.

        Equivalent to Rust's Result::unwrap_err().
        """
        ...

    def map_or_else[U](self, ok: Callable[[T], U], err: Callable[[E], U]) -> U:
        """
        Pattern matches on the result, calling ok if Ok, or err if Err.

        Args:
            ok: Callable to handle the Ok value.
            err: Callable to handle the Err value.

        Returns:
            The result of the called function.

        Equivalent to Rust's Result::map_or_else()
        """
        match self.is_ok():
            case True:
                return ok(self.unwrap())
            case False:
                return err(self.unwrap_err())
            case _:
                raise RuntimeError("unreachable")

    def expect(self, msg: str) -> T:
        """
        Returns the contained Ok value, or raises ResultUnwrapError with a custom message if the result is Err.

        Args:
            msg: The message to display if the result is Err.

        Returns:
            The contained Ok value.

        Raises:
            ResultUnwrapError: If the result is Err, with the provided message and error.

        Equivalent to Rust's Result::expect().
        """
        if self.is_ok():
            return self.unwrap()
        raise ResultUnwrapError(f"{msg}: {self.unwrap_err()}")

    def expect_err(self, msg: str) -> E:
        """
        Returns the contained Err value, or raises ResultUnwrapError with a custom message if the result is Ok.

        Args:
            msg: The message to display if the result is Ok.

        Returns:
            The contained Err value.

        Raises:
            ResultUnwrapError: If the result is Ok, with the provided message and value.

        Equivalent to Rust's Result::expect_err().
        """
        if self.is_err():
            return self.unwrap_err()
        raise ResultUnwrapError(f"{msg}: expected Err, got Ok({self.unwrap()!r})")

    def unwrap_or(self, default: T) -> T:
        """
        Returns the contained Ok value or a provided default.

        Args:
            default: The value to return if the result is Err.

        Returns:
            The contained Ok value or the default.

        Equivalent to Rust's Result::unwrap_or().
        """
        return self.unwrap() if self.is_ok() else default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """
        Returns the contained Ok value or computes it from a function if Err.

        Args:
            f: Callable that takes the Err value and returns a T.

        Returns:
            The contained Ok value or the result of f(error).

        Equivalent to Rust's Result::unwrap_or_else().
        """
        return self.unwrap() if self.is_ok() else f(self.unwrap_err())

    def map[U](self, f: Callable[[T], U]) -> Result[U, E]:
        """
        Maps a Result[T, E] to Result[U, E] by applying a function to a contained Ok value, leaving Err untouched.

        Args:
            f: Callable to apply to the Ok value.

        Returns:
            Result[U, E]: Ok(f(value)) if Ok, otherwise Err(error).

        Equivalent to Rust's Result::map().
        """
        if self.is_ok():
            return Ok(f(self.unwrap()))
        return cast(Result[U, E], self)

    def map_err[F](self, f: Callable[[E], F]) -> Result[T, F]:
        """
        Maps a Result[T, E] to Result[T, F] by applying a function to a contained Err value, leaving Ok untouched.

        Args:
            f: Callable to apply to the Err value.

        Returns:
            Result[T, F]: Err(f(error)) if Err, otherwise Ok(value).

        Equivalent to Rust's Result::map_err().
        """
        if self.is_err():
            return Err(f(self.unwrap_err()))
        return cast(Result[T, F], self)

    def and_then[U](self, f: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """
        Calls f if the result is Ok, otherwise returns Err.

        Args:
            f: Callable that takes the Ok value and returns a Result.

        Returns:
            Result[U, E]: The result of f(value) if Ok, otherwise Err(error).

        Equivalent to Rust's Result::and_then().
        """
        if self.is_ok():
            return f(self.unwrap())
        return cast(Result[U, E], self)

    def or_else(self, f: Callable[[E], Result[T, E]]) -> Result[T, E]:
        """
        Calls f if the result is Err, otherwise returns Ok.

        Args:
            f: Callable that takes the Err value and returns a Result.

        Returns:
            Result[T, E]: self if Ok, otherwise the result of f(error).

        Equivalent to Rust's Result::or_else().
        """
        return self if self.is_ok() else f(self.unwrap_err())

    def ok(self) -> Option[T]:
        """
        Converts the Result into an Option, mapping Ok(v) to Some(v) and Err(e) to None.

        Returns:
            Option[T]: Some(value) if Ok, otherwise None.
        Equivalent to Rust's Result::ok().
        """
        if self.is_ok():
            return Some(self.unwrap())
        return NONE

    def err(self) -> Option[E]:
        """
        Converts the Result into an Option, mapping Err(e) to Some(e) and Ok(v) to None.

        Returns:
            Option[E]: Some(error) if Err, otherwise None.
        Equivalent to Rust's Result::err().
        """
        if self.is_err():
            return Some(self.unwrap_err())
        return NONE


@dataclass(slots=True)
class Ok[T, E](Result[T, E]):
    value: T

    def is_ok(self) -> TypeIs[Ok[T, E]]:  # type: ignore[misc]
        """
        Always returns True for Ok.

        Equivalent to Rust's Result::is_ok().
        """
        return True

    def is_err(self) -> TypeIs[Err[T, E]]:  # type: ignore[misc]
        """
        Always returns False for Ok.

        Equivalent to Rust's Result::is_err().
        """
        return False

    def unwrap(self) -> T:
        """
        Returns the contained Ok value.

        Returns:
            The contained value.

        Equivalent to Rust's Result::unwrap() for Ok.
        """
        return self.value

    def unwrap_err(self) -> Never:
        """
        Raises ResultUnwrapError because there is no error value.

        Raises:
            ResultUnwrapError: Always, since Ok contains no error.

        Equivalent to Rust's Result::unwrap_err() for Ok.
        """
        raise ResultUnwrapError("called `unwrap_err` on Ok")


@dataclass(slots=True)
class Err[T, E](Result[T, E]):
    error: E

    def is_ok(self) -> TypeIs[Ok[T, E]]:  # type: ignore[misc]
        """
        Always returns False for Err.

        Equivalent to Rust's Result::is_ok().
        """
        return False

    def is_err(self) -> TypeIs[Err[T, E]]:  # type: ignore[misc]
        """
        Always returns True for Err.

        Equivalent to Rust's Result::is_err().
        """
        return True

    def unwrap(self) -> Never:
        """
        Raises ResultUnwrapError because there is no Ok value.

        Raises:
            ResultUnwrapError: Always, since Err contains no value.

        Equivalent to Rust's Result::unwrap() for Err.
        """
        raise ResultUnwrapError(f"called `unwrap` on Err: {self.error!r}")

    def unwrap_err(self) -> E:
        """
        Returns the contained error value.

        Returns:
            The contained error value.

        Equivalent to Rust's Result::unwrap_err() for Err.
        """
        return self.error


class Wrapper[T](CommonBase[T]):
    """
    A generic Wrapper for any type.
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
