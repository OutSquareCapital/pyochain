from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Concatenate, Never, cast

import cytoolz as cz

from ._core import Pipeable

if TYPE_CHECKING:
    from ._iter import Iter
    from ._option import Option


class ResultUnwrapError(RuntimeError): ...


class Result[T, E](Pipeable, ABC):
    """`Result[T, E]` is the type used for returning and propagating errors.

    It is a class that can represent two variants, `Ok[T]`, representing success and containing a value, and `Err[E]`, representing error and containing an error value.

    Functions return `Result` whenever errors are expected and recoverable.

    For example, I/O or web requests can fail for many reasons, and using `Result` forces the caller to handle the possibility of failure.

    This is directly inspired by Rust's `Result` type, and provides similar functionality for error handling in Python.

    """

    def flatten(self: Result[Result[T, E], E]) -> Result[T, E]:
        """Flattens a nested `Result`.

        Converts from `Result[Result[T, E], E]` to `Result[T, E]`.

        Equivalent to calling `Result.and_then(lambda x: x)`, but more convenient when there's no need to process the inner `Ok` value.

        Returns:
            Result[T, E]: The flattened result.

        Example:
        ```python
        >>> import pyochain as pc
        >>> nested_ok: pc.Result[pc.Result[int, str], str] = pc.Ok(pc.Ok(2))
        >>> nested_ok.flatten()
        Ok(2)
        >>> nested_err: pc.Result[pc.Result[int, str], str] = pc.Ok(pc.Err("inner error"))
        >>> nested_err.flatten()
        Err('inner error')

        ```
        """
        return self.and_then(cz.functoolz.identity)

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
        pyochain._result.ResultUnwrapError: called `unwrap` on Err: 'emergency failure'

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
        pyochain._result.ResultUnwrapError: called `unwrap_err` on Ok

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
        pyochain._result.ResultUnwrapError: Testing expect: emergency failure

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
        pyochain._result.ResultUnwrapError: Testing expect_err: expected Err, got Ok(10)

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

    def unwrap_or_else[**P](
        self, fn: Callable[Concatenate[E, P], T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Returns the contained `Ok` value or computes it from a function.

        Args:
            fn (Callable[Concatenate[E, P], T]): A function that takes the `Err` value and returns a default value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

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
        return self.unwrap() if self.is_ok() else fn(self.unwrap_err(), *args, **kwargs)

    def map[**P, R](
        self, fn: Callable[Concatenate[T, P], R], *args: P.args, **kwargs: P.kwargs
    ) -> Result[R, E]:
        """Maps a `Result[T, E]` to `Result[U, E]`.

        Done by applying a function to a contained `Ok` value,
        leaving an `Err` value untouched.

        Args:
            fn (Callable[Concatenate[T, P], R]): The function to apply to the `Ok` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[R, E]: A new `Result` with the mapped value if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map(lambda x: x * 2)
        Ok(4)
        >>> pc.Err("error").map(lambda x: x * 2)
        Err('error')

        ```
        """
        return (
            Ok(fn(self.unwrap(), *args, **kwargs))
            if self.is_ok()
            else cast(Result[R, E], self)
        )

    def map_err[**P, R](
        self, fn: Callable[Concatenate[E, P], R], *args: P.args, **kwargs: P.kwargs
    ) -> Result[T, R]:
        """Maps a `Result[T, E]` to `Result[T, R]`.

        Done by applying a function to a contained `Err` value,
        leaving an `Ok` value untouched.

        Args:
            fn (Callable[Concatenate[E, P], R]): The function to apply to the `Err` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.


        Returns:
            Result[T, R]: A new `Result` with the mapped error if `Err`, otherwise the original `Ok`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map_err(len)
        Ok(2)
        >>> pc.Err("foo").map_err(len)
        Err(3)

        ```
        """
        return (
            Err(fn(self.unwrap_err(), *args, **kwargs))
            if self.is_err()
            else cast(Result[T, R], self)
        )

    def inspect[**P](
        self, fn: Callable[Concatenate[T, P], object], *args: P.args, **kwargs: P.kwargs
    ) -> Result[T, E]:
        """Applies a function to the contained `Ok` value, returning the original `Result`.

        This is primarily useful for debugging or logging, allowing side effects to be
        performed on the `Ok` value without changing the result.

        Args:
            fn (Callable[Concatenate[T, P], object]): Function to apply to the `Ok` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[T, E]: The original result, unchanged.

        Example:
        ```python
        >>> import pyochain as pc
        >>> seen: list[int] = []
        >>> pc.Ok(2).inspect(lambda x: seen.append(x))
        Ok(2)
        >>> seen
        [2]

        ```
        """
        if self.is_ok():
            fn(self.unwrap(), *args, **kwargs)
        return self

    def inspect_err[**P](
        self, fn: Callable[Concatenate[E, P], object], *args: P.args, **kwargs: P.kwargs
    ) -> Result[T, E]:
        """Applies a function to the contained `Err` value, returning the original `Result`.

        This mirrors :meth:`inspect` but operates on the error value. It is useful for
        logging or debugging error paths while keeping the `Result` unchanged.

        Args:
            fn (Callable[Concatenate[E, P], object]): Function to apply to the `Err` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[T, E]: The original result, unchanged.

        Example:
        ```python
        >>> import pyochain as pc
        >>> seen: list[str] = []
        >>> pc.Err("oops").inspect_err(lambda e: seen.append(e))
        Err('oops')
        >>> seen
        ['oops']

        ```
        """
        if self.is_err():
            fn(self.unwrap_err(), *args, **kwargs)
        return self

    def and_[U](self, res: Result[U, E]) -> Result[U, E]:
        """Returns `res` if the result is `Ok`, otherwise returns the `Err` value.

        This is often used for chaining operations that might fail.

        Args:
            res (Result[U, E]): The result to return if the original result is `Ok`.

        Returns:
            Result[U, E]: `res` if the original result is `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x = pc.Ok(2)
        >>> y = pc.Err("late error")
        >>> x.and_(y)
        Err('late error')
        >>> x = pc.Err("early error")
        >>> y = pc.Ok("foo")
        >>> x.and_(y)
        Err('early error')

        >>> x = pc.Err("not a 2")
        >>> y = pc.Err("late error")
        >>> x.and_(y)
        Err('not a 2')

        >>> x = pc.Ok(2)
        >>> y = pc.Ok("different result type")
        >>> x.and_(y)
        Ok('different result type')

        ```
        """
        return res if self.is_ok() else cast(Result[U, E], self)

    def and_then[**P, R](
        self,
        fn: Callable[Concatenate[T, P], Result[R, E]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[R, E]:
        """Calls a function if the result is `Ok`, otherwise returns the `Err` value.

        This is often used for chaining operations that might fail.

        Args:
            fn (Callable[Concatenate[T, P], Result[R, E]]): The function to call with the `Ok` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[R, E]: The result of the function if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def to_str(x: int) -> Result[str, str]:
        ...     return pc.Ok(str(x))
        >>> pc.Ok(2).and_then(to_str)
        Ok('2')
        >>> pc.Err("error").and_then(to_str)
        Err('error')

        ```
        """
        return (
            fn(self.unwrap(), *args, **kwargs)
            if self.is_ok()
            else cast(Result[R, E], self)
        )

    def or_else[**P](
        self,
        fn: Callable[Concatenate[E, P], Result[T, E]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[T, E]:
        """Calls a function if the result is `Err`, otherwise returns the `Ok` value.

        This is often used for handling errors by trying an alternative operation.

        Args:
            fn (Callable[Concatenate[E, P], Result[T, E]]): The function to call with the `Err` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[T, E]: The original `Ok` value, or the result of the function if `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def fallback(e: str) -> Result[int, str]:
        ...     return pc.Ok(len(e))
        >>> pc.Ok(2).or_else(fallback)
        Ok(2)
        >>> pc.Err("foo").or_else(fallback)
        Ok(3)

        ```
        """
        return self if self.is_ok() else fn(self.unwrap_err(), *args, **kwargs)

    def ok(self) -> Option[T]:
        """Converts from `Result[T, E]` to `Option[T]`.

        `Ok(v)` becomes `Some(v)`, and `Err(e)` becomes `None`.

        Returns:
            Option[T]: An `Option` containing the `Ok` value, or `None` if the result is `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).ok()
        Some(2)
        >>> pc.Err("error").ok()
        NONE

        ```
        """
        from ._option import NONE, Some

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
        Some('error')

        ```
        """
        from ._option import NONE, Some

        return Some(self.unwrap_err()) if self.is_err() else NONE

    def is_ok_and[**P](
        self, pred: Callable[Concatenate[T, P], bool], *args: P.args, **kwargs: P.kwargs
    ) -> bool:
        """Returns True if the result is `Ok` and the predicate is true for the contained value.

        Args:
            pred (Callable[Concatenate[T, P], bool]): Predicate function to apply to the `Ok` value.
            *args (P.args): Additional positional arguments to pass to pred.
            **kwargs (P.kwargs): Additional keyword arguments to pass to pred.

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
        return self.is_ok() and pred(self.unwrap(), *args, **kwargs)

    def is_err_and[**P](
        self, pred: Callable[Concatenate[E, P], bool], *args: P.args, **kwargs: P.kwargs
    ) -> bool:
        """Returns True if the result is Err and the predicate is true for the error value.

        Args:
            pred (Callable[Concatenate[E, P], bool]): Predicate function to apply to the Err value.
            *args (P.args): Additional positional arguments to pass to pred.
            **kwargs (P.kwargs): Additional keyword arguments to pass to pred.

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
        return self.is_err() and pred(self.unwrap_err(), *args, **kwargs)

    def map_or[**P, R](
        self,
        default: R,
        f: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Applies a function to the `Ok` value if present, otherwise returns the default value.

        Args:
            default (R): Value to return if the result is Err.
            f (Callable[Concatenate[T, P], R]): Function to apply to the `Ok` value.
            *args (P.args): Additional positional arguments to pass to f.
            **kwargs (P.kwargs): Additional keyword arguments to pass to f.

        Returns:
            R: Result of f(value) if Ok, otherwise default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).map_or(10, lambda x: x * 2)
        4
        >>> pc.Err("err").map_or(10, lambda x: x * 2)
        10

        ```
        """
        return f(self.unwrap(), *args, **kwargs) if self.is_ok() else default

    def iter(self) -> Iter[T]:
        """Returns a `Iter[T]` over the possibly contained value.

        The iterator yields one value if the result is `Ok`, otherwise none.

        Returns:
            Iter[T]: An iterator over the `Ok` value, or empty if `Err`.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(7).iter().next()
        Some(7)
        >>> pc.Err("nothing!").iter().next()
        NONE

        ```
        """
        return self.ok().iter()

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
        Some(Ok(2))
        >>> pc.Ok(pc.NONE).transpose()
        NONE
        >>> pc.Err("err").transpose()
        Some(Err('err'))

        ```
        """
        from ._option import NONE, Some

        if self.is_err():
            return Some(Err(self.unwrap_err()))
        opt = self.unwrap()
        if opt.is_none():
            return NONE
        return Some(Ok(opt.unwrap()))

    def or_[F](self, res: Result[T, F]) -> Result[T, F]:
        """Returns res if the result is `Err`, otherwise returns the `Ok` value of **self**.

        Args:
            res (Result[T, F]): The result to return if the original result is `Err`.

        Returns:
            Result[T, F]: The original `Ok` value, or `res` if the original result is `Err`.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(2).or_(pc.Err("late error"))
        Ok(2)
        >>> pc.Err("early error").or_(pc.Ok(2))
        Ok(2)
        >>> pc.Err("not a 2").or_(pc.Err("late error"))
        Err('late error')
        >>> pc.Ok(2).or_(pc.Ok(100))
        Ok(2)

        ```
        """
        return cast(Result[T, F], self) if self.is_ok() else res


@dataclass(slots=True)
class Ok[T, E](Result[T, E]):
    """Represents a successful value.

    Attributes:
        value (T): The contained successful value.
    """

    value: T

    def __repr__(self) -> str:
        return f"Ok({self.value!r})"

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

    def __repr__(self) -> str:
        return f"Err({self.error!r})"

    def is_ok(self) -> bool:
        return False

    def is_err(self) -> bool:
        return True

    def unwrap(self) -> Never:
        msg = f"called `unwrap` on Err: {self.error!r}"
        raise ResultUnwrapError(msg)

    def unwrap_err(self) -> E:
        return self.error
