from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, cast

from .._core import Pipeable

if TYPE_CHECKING:
    from ._states import Option


class ResultUnwrapError(RuntimeError): ...


class Result[T, E](Pipeable, ABC):
    """`Result[T, E]` is the type used for returning and propagating errors.

    It is a class that can represent two variants, `Ok[T]`, representing success and containing a value, and `Err[E]`, representing error and containing an error value.

    Functions return `Result` whenever errors are expected and recoverable.

    For example, I/O or web requests can fail for many reasons, and using `Result` forces the caller to handle the possibility of failure.

    This is directly inspired by Rust's `Result` type, and provides similar functionality for error handling in Python.

    """

    @staticmethod
    def from_fn[U, F](
        *exc_types: type[BaseException],
        fn: Callable[[], U],
        map_err: Callable[[BaseException], F] = lambda e: e,
    ) -> Result[U, F]:
        """Creates a `Result` from a function that may raise exceptions.

        All exceptions of the specified types are caught and mapped to `Err` using the optional provided mapping function.

        Args:
            *exc_types (type[BaseException]): Exception types to catch.
            fn (Callable[[], U]): The function to execute.
            map_err (Callable[[BaseException], F]): A function that maps caught exceptions to error values. Defaults to identity function.

        Returns:
            Result[U, F]: `Ok` with the function's return value if no exception is raised, otherwise `Err` with the mapped error.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def might_fail() -> int:
        ...     return 1 / 0
        >>> pc.Result.from_fn(ZeroDivisionError, fn=might_fail, map_err=str)
        Err(error='division by zero')

        ```
        """
        from ._states import Err, Ok

        try:
            return Ok(fn())
        except exc_types as exc:
            return Err(map_err(exc))

    @abstractmethod
    def is_ok(self) -> bool:
        """Returns `True` if the result is `Ok`.

        Returns:
            bool: `True` if the result is an `Ok` variant, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x: pc.Result[int, str] = pc.Ok(2)
        >>> x.is_ok()
        True
        >>> y: pc.Result[int, str] = pc.Err("Some error message")
        >>> y.is_ok()
        False

        ```

        """
        ...

    @abstractmethod
    def is_err(self) -> bool:
        """Returns `True` if the result is `Err`.

        Returns:
            bool: `True` if the result is an `Err` variant, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x: pc.Result[int, str] = pc.Ok(2)
        >>> x.is_err()
        False
        >>> y: pc.Result[int, str] = pc.Err("Some error message")
        >>> y.is_err()
        True

        ```

        """
        ...

    @abstractmethod
    def unwrap(self) -> T:
        """Returns the contained `Ok` value.

        Similar to `Iter.inner()`, this is a terminal operation that ends the chain.

        However, since this can raise an exception if the result is `Err`, the terminology is kept consistent with Rust's `Result` type.

        Returns:
            T: The contained `Ok` value.

        Raises:
            ResultUnwrapError: If the result is `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).unwrap()
        2

        ```
        ```python
        >>> import pyochain as pc
        >>> pc.Err("emergency failure").unwrap()
        Traceback (most recent call last):
            ...
        pyochain._results._result.ResultUnwrapError: called `unwrap` on Err: 'emergency failure'

        ```

        """
        ...

    @abstractmethod
    def unwrap_err(self) -> E:
        """Returns the contained `Err` value.

        Returns:
            E: The contained `Err` value.

        Raises:
            ResultUnwrapError: If the result is `Ok`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Err("emergency failure").unwrap_err()
        'emergency failure'

        ```
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).unwrap_err()
        Traceback (most recent call last):
            ...
        pyochain._results._result.ResultUnwrapError: called `unwrap_err` on Ok

        ```

        """
        ...

    def map_or_else[U](self, ok: Callable[[T], U], err: Callable[[E], U]) -> U:
        """Maps a `Result[T, E]` to `U`.

        Done by applying a fallback function to a contained `Err` value,
        or a default function to a contained `Ok` value.

        Args:
            ok (Callable[[T], U]): The function to apply to the `Ok` value.
            err (Callable[[E], U]): The function to apply to the `Err` value.

        Returns:
            U: The result of applying the appropriate function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> k = 21
        >>> pc.Ok("foo").map_or_else(len, lambda e: k * 2)
        3
        >>> pc.Err("bar").map_or_else(len, lambda e: k * 2)
        42

        ```

        """
        return ok(self.unwrap()) if self.is_ok() else err(self.unwrap_err())

    def expect(self, msg: str) -> T:
        """Returns the contained `Ok` value.

        Raises an exception with a provided message if the value is an `Err`.

        Args:
            msg (str): The message to include in the exception if the result is `Err`.

        Returns:
            T: The contained `Ok` value.

        Raises:
            ResultUnwrapError: If the result is `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).expect("No error")
        2
        >>> pc.Err("emergency failure").expect("Testing expect")
        Traceback (most recent call last):
            ...
        pyochain._results._result.ResultUnwrapError: Testing expect: emergency failure

        ```

        """
        if self.is_ok():
            return self.unwrap()
        err_msg = f"{msg}: {self.unwrap_err()}"
        raise ResultUnwrapError(err_msg)

    def expect_err(self, msg: str) -> E:
        """Returns the contained `Err` value.

        Raises an exception with a provided message if the value is an `Ok`.

        Args:
            msg (str): The message to include in the exception if the result is `Ok`.

        Returns:
            E: The contained `Err` value.

        Raises:
            ResultUnwrapError: If the result is `Ok`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Err("emergency failure").expect_err("Testing expect_err")
        'emergency failure'
        >>> pc.Ok(10).expect_err("Testing expect_err")
        Traceback (most recent call last):
            ...
        pyochain._results._result.ResultUnwrapError: Testing expect_err: expected Err, got Ok(10)

        ```

        """
        if self.is_err():
            return self.unwrap_err()
        err_msg = f"{msg}: expected Err, got Ok({self.unwrap()!r})"
        raise ResultUnwrapError(err_msg)

    def unwrap_or(self, default: T) -> T:
        """Returns the contained `Ok` value or a provided default.

        Args:
            default (T): The value to return if the result is `Err`.

        Returns:
            T: The contained `Ok` value or the provided default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).unwrap_or(10)
        2
        >>> pc.Err("error").unwrap_or(10)
        10

        ```

        """
        return self.unwrap() if self.is_ok() else default

    def unwrap_or_else(self, fn: Callable[[E], T]) -> T:
        """Returns the contained `Ok` value or computes it from a function.

        Args:
            fn (Callable[[E], T]): A function that takes the `Err` value and returns a default value.

        Returns:
            T: The contained `Ok` value or the result of the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).unwrap_or_else(len)
        2
        >>> pc.Err("foo").unwrap_or_else(len)
        3

        ```

        """
        return self.unwrap() if self.is_ok() else fn(self.unwrap_err())

    def map[U](self, fn: Callable[[T], U]) -> Result[U, E]:
        """Maps a `Result[T, E]` to `Result[U, E]`.

        Done by applying a function to a contained `Ok` value,
        leaving an `Err` value untouched.

        Args:
            fn (Callable[[T], U]): The function to apply to the `Ok` value.

        Returns:
            Result[U, E]: A new `Result` with the mapped value if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map(lambda x: x * 2)
        Ok(value=4)
        >>> pc.Err("error").map(lambda x: x * 2)
        Err(error='error')

        ```

        """
        from ._states import Ok

        return Ok(fn(self.unwrap())) if self.is_ok() else cast(Result[U, E], self)

    def map_err[F](self, fn: Callable[[E], F]) -> Result[T, F]:
        """Maps a `Result[T, E]` to `Result[T, F]`.

        Done by applying a function to a contained `Err` value,
        leaving an `Ok` value untouched.

        Args:
            fn (Callable[[E], F]): The function to apply to the `Err` value.

        Returns:
            Result[T, F]: A new `Result` with the mapped error if `Err`, otherwise the original `Ok`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map_err(len)
        Ok(value=2)
        >>> pc.Err("foo").map_err(len)
        Err(error=3)

        ```

        """
        from ._states import Err

        return Err(fn(self.unwrap_err())) if self.is_err() else cast(Result[T, F], self)

    def and_then[U](self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Calls a function if the result is `Ok`, otherwise returns the `Err` value.

        This is often used for chaining operations that might fail.

        Args:
            fn (Callable[[T], Result[U, E]]): The function to call with the `Ok` value.

        Returns:
            Result[U, E]: The result of the function if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def to_str(x: int) -> Result[str, str]:
        ...     return pc.Ok(str(x))
        >>> pc.Ok(2).and_then(to_str)
        Ok(value='2')
        >>> pc.Err("error").and_then(to_str)
        Err(error='error')

        ```

        """
        return fn(self.unwrap()) if self.is_ok() else cast(Result[U, E], self)

    def or_else(self, fn: Callable[[E], Result[T, E]]) -> Result[T, E]:
        """Calls a function if the result is `Err`, otherwise returns the `Ok` value.

        This is often used for handling errors by trying an alternative operation.

        Args:
            fn (Callable[[E], Result[T, E]]): The function to call with the `Err` value.

        Returns:
            Result[T, E]: The original `Ok` value, or the result of the function if `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def fallback(e: str) -> Result[int, str]:
        ...     return pc.Ok(len(e))
        >>> pc.Ok(2).or_else(fallback)
        Ok(value=2)
        >>> pc.Err("foo").or_else(fallback)
        Ok(value=3)

        ```

        """
        return self if self.is_ok() else fn(self.unwrap_err())

    def ok(self) -> Option[T]:
        """Converts from `Result[T, E]` to `Option[T]`.

        `Ok(v)` becomes `Some(v)`, and `Err(e)` becomes `None`.

        Returns:
            Option[T]: An `Option` containing the `Ok` value, or `None` if the result is `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).ok()
        Some(value=2)
        >>> pc.Err("error").ok()
        NONE

        ```

        """
        from ._states import NONE, Some

        return Some(self.unwrap()) if self.is_ok() else NONE

    def err(self) -> Option[E]:
        """Converts from `Result[T, E]` to `Option[E]`.

        `Err(e)` becomes `Some(e)`, and `Ok(v)` becomes `None`.

        Returns:
            Option[E]: An `Option` containing the `Err` value, or `None` if the result is `Ok`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).err()
        NONE
        >>> pc.Err("error").err()
        Some(value='error')

        ```

        """
        from ._states import NONE, Some

        return Some(self.unwrap_err()) if self.is_err() else NONE

    def is_ok_and(self, pred: Callable[[T], bool]) -> bool:
        """Returns True if the result is `Ok` and the predicate is true for the contained value.

        Args:
            pred (Callable[[T], bool]): Predicate function to apply to the `Ok` value.

        Returns:
            bool: True if `Ok` and pred(value) is true, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).is_ok_and(lambda x: x > 1)
        True
        >>> pc.Ok(0).is_ok_and(lambda x: x > 1)
        False
        >>> pc.Err("err").is_ok_and(lambda x: x > 1)
        False

        ```

        """
        return self.is_ok() and pred(self.unwrap())

    def is_err_and(self, pred: Callable[[E], bool]) -> bool:
        """Returns True if the result is Err and the predicate is true for the error value.

        Args:
            pred (Callable[[E], bool]): Predicate function to apply to the Err value.

        Returns:
            bool: True if Err and pred(error) is true, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Err("foo").is_err_and(lambda e: len(e) == 3)
        True
        >>> pc.Err("bar").is_err_and(lambda e: e == "baz")
        False
        >>> pc.Ok(2).is_err_and(lambda e: True)
        False

        ```

        """
        return self.is_err() and pred(self.unwrap_err())

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        """Applies a function to the `Ok` value if present, otherwise returns the default value.

        Args:
            default (U): Value to return if the result is Err.
            f (Callable[[T], U]): Function to apply to the `Ok` value.

        Returns:
            U: Result of f(value) if Ok, otherwise default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map_or(10, lambda x: x * 2)
        4
        >>> pc.Err("err").map_or(10, lambda x: x * 2)
        10

        ```

        """
        return f(self.unwrap()) if self.is_ok() else default

    def transpose(self: Result[Option[T], E]) -> Option[Result[T, E]]:
        """Transposes a Result containing an Option into an Option containing a Result.

        Can only be called if the inner type is `Option[T, E]`.

        `Ok(Some(v)) -> Some(Ok(v)), Ok(NONE) -> NONE, Err(e) -> Some(Err(e))`

        Returns:
            Option[Result[T, E]]: Option containing a Result or NONE.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(pc.Some(2)).transpose()
        Some(value=Ok(value=2))
        >>> pc.Ok(pc.NONE).transpose()
        NONE
        >>> pc.Err("err").transpose()
        Some(value=Err(error='err'))

        ```

        """
        from ._states import NONE, Err, Ok, Some

        if self.is_err():
            return Some(Err(self.unwrap_err()))
        opt = self.unwrap()
        if opt.is_none():
            return NONE
        return Some(Ok(opt.unwrap()))
