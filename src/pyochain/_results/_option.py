from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Never, TypeIs

if TYPE_CHECKING:
    from ._result import Result


class OptionUnwrapError(RuntimeError): ...


class Option[T](ABC):
    @abstractmethod
    def is_some(self) -> TypeIs[Some[T]]:  # type: ignore[misc]
        """
        Returns `True` if the option is a `Some` value.

        Uses `TypeIs[Some[T]]` for more precise type narrowing.

        Returns:
            bool: `True` if the option is a `Some` variant, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x: Option[int] = pc.Some(2)
        >>> x.is_some()
        True
        >>> y: Option[int] = pc.NONE
        >>> y.is_some()
        False

        ```
        """
        ...

    @abstractmethod
    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """
        Returns `True` if the option is a `None` value.

        Uses `TypeIs[_None]` for more precise type narrowing.

        Returns:
            bool: `True` if the option is a `_None` variant, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x: Option[int] = pc.Some(2)
        >>> x.is_none()
        False
        >>> y: Option[int] = pc.NONE
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
            T: The contained `Some` value.

        Raises:
            OptionUnwrapError: If the option is `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some("car").unwrap()
        'car'

        ```
        ```python
        >>> import pyochain as pc
        >>> pc.NONE.unwrap()
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
            msg (str): The message to include in the exception if the result is `None`.

        Returns:
            T: The contained `Some` value.

        Raises:
            OptionUnwrapError: If the result is `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some("value").expect("fruits are healthy")
        'value'
        >>> pc.NONE.expect("fruits are healthy")
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
            default (T): The value to return if the result is `None`.

        Returns:
            T: The contained `Some` value or the provided default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some("car").unwrap_or("bike")
        'car'
        >>> pc.NONE.unwrap_or("bike")
        'bike'

        ```
        """
        return self.unwrap() if self.is_some() else default

    def unwrap_or_else(self, f: Callable[[], T]) -> T:
        """
        Returns the contained `Some` value or computes it from a function.

        Args:
            f (Callable[[], T]): A function that returns a default value if the result is `None`.

        Returns:
            T: The contained `Some` value or the result of the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> k = 10
        >>> pc.Some(4).unwrap_or_else(lambda: 2 * k)
        4
        >>> pc.NONE.unwrap_or_else(lambda: 2 * k)
        20

        ```
        """
        return self.unwrap() if self.is_some() else f()

    def map[U](self, f: Callable[[T], U]) -> Option[U]:
        """
        Maps an `Option[T]` to `Option[U]` by applying a function to a contained `Some` value,
        leaving a `None` value untouched.

        Args:
            f (Callable[[T], U]): The function to apply to the `Some` value.

        Returns:
            Option[U]: A new `Option` with the mapped value if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some("Hello, World!").map(len)
        Some(value=13)
        >>> pc.NONE.map(len)
        NONE

        ```
        """
        return Some(f(self.unwrap())) if self.is_some() else NONE

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """
        Calls a function if the option is `Some`, otherwise returns `None`.

        Args:
            f (Callable[[T], Option[U]]): The function to call with the `Some` value.

        Returns:
            Option[U]: The result of the function if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def sq(x: int) -> Option[int]:
        ...     return Some(x * x)
        >>> def nope(x: int) -> Option[int]:
        ...     return pc.NONE
        >>> pc.Some(2).and_then(sq).and_then(sq)
        Some(value=16)
        >>> pc.Some(2).and_then(sq).and_then(nope)
        NONE
        >>> pc.Some(2).and_then(nope).and_then(sq)
        NONE
        >>> pc.NONE.and_then(sq).and_then(sq)
        NONE

        ```
        """
        return f(self.unwrap()) if self.is_some() else NONE

    def or_else(self, f: Callable[[], Option[T]]) -> Option[T]:
        """
        Returns the option if it contains a value, otherwise calls a function and returns the result.

        Args:
            f (Callable[[], Option[T]]): The function to call if the option is `None`.

        Returns:
            Option[T]: The original `Option` if it is `Some`, otherwise the result of the function.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def nobody() -> Option[str]:
        ...     return pc.NONE
        >>> def vikings() -> Option[str]:
        ...     return Some("vikings")
        >>> pc.Some("barbarians").or_else(vikings)
        Some(value='barbarians')
        >>> pc.NONE.or_else(vikings)
        Some(value='vikings')
        >>> pc.NONE.or_else(nobody)
        NONE

        ```
        """
        return self if self.is_some() else f()

    def zip[U](self, other: Option[U]) -> Option[tuple[T, U]]:
        """
        Returns an Option containing a tuple of the values if both options are Some, otherwise returns NONE.

        Args:
            other (Option[U]): The other option to zip with.

        Returns:
            Option[tuple[T, U]]: Some((self, other)) if both are Some, otherwise NONE.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(1).zip(pc.Some('a'))
        Some(value=(1, 'a'))
        >>> pc.Some(1).zip(pc.NONE)
        NONE
        >>> pc.NONE.zip(pc.Some('a'))
        NONE

        ```
        """
        if self.is_some() and other.is_some():
            return Some((self.unwrap(), other.unwrap()))
        return NONE

    def zip_with[U, R](self, other: Option[U], f: Callable[[T, U], R]) -> Option[R]:
        """

        Zips `self` and another `Option` with function `f`.

        If `self` is `Some(s)` and other is `Some(o)`, this method returns `Some(f(s, o))`.

        Otherwise, `NONE` is returned.

        Args:
            other (pc.Option[U]): The second option.
            f (Callable[[T, U], R]): The function to apply to the unwrapped values.
        Returns:
            pc.Option[R]: The resulting option after applying the function.
        Examples:
        ```python
        >>> from dataclasses import dataclass
        >>> import pyochain as pc
        >>>
        >>> @dataclass
        ... class Point:
        ...     x: float
        ...     y: float
        >>>
        >>> x = pc.Some(17.5)
        >>> y = pc.Some(42.7)
        >>> x.zip_with(y, Point)
        Some(value=Point(x=17.5, y=42.7))
        >>> x.zip_with(pc.NONE, Point)
        NONE

        ```
        """
        if self.is_some() and other.is_some():
            return Some(f(self.unwrap(), other.unwrap()))
        return NONE

    def ok_or[E](self, err: E) -> Result[T, E]:
        """
        Converts the option to a Result, mapping Some(v) to Ok(v) and NONE to Err(err).

        Args:
            err (E): The error value to use if the option is NONE.

        Returns:
            Result[T, E]: Ok(v) if Some(v), otherwise Err(err).

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(1).ok_or('fail')
        Ok(value=1)
        >>> pc.NONE.ok_or('fail')
        Err(error='fail')

        ```
        """
        from ._result import Err, Ok

        return Ok(self.unwrap()) if self.is_some() else Err(err)

    def ok_or_else[E](self, err: Callable[[], E]) -> Result[T, E]:
        """
        Converts the option to a Result, mapping Some(v) to Ok(v) and NONE to Err(err()).

        Args:
            err (Callable[[], E]): A function returning the error value if the option is NONE.

        Returns:
            Result[T, E]: Ok(v) if Some(v), otherwise Err(err()).

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(1).ok_or_else(lambda: 'fail')
        Ok(value=1)
        >>> pc.NONE.ok_or_else(lambda: 'fail')
        Err(error='fail')

        ```
        """
        from ._result import Err, Ok

        return Ok(self.unwrap()) if self.is_some() else Err(err())

    def map_or[U](self, default: U, f: Callable[[T], U]) -> U:
        """
        Returns the result of applying a function to the contained value if Some, otherwise returns the default value.

        Args:
            default (U): The default value to return if NONE.
            f (Callable[[T], U]): The function to apply to the contained value.

        Returns:
            U: The result of f(self.unwrap()) if Some, otherwise default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).map_or(0, lambda x: x * 10)
        20
        >>> pc.NONE.map_or(0, lambda x: x * 10)
        0

        ```
        """
        return f(self.unwrap()) if self.is_some() else default

    def map_or_else[U](self, default: Callable[[], U], f: Callable[[T], U]) -> U:
        """
        Returns the result of applying a function to the contained value if Some, otherwise computes a default value.

        Args:
            default (Callable[[], U]): A function returning the default value if NONE.
            f (Callable[[T], U]): The function to apply to the contained value.

        Returns:
            U: The result of f(self.unwrap()) if Some, otherwise default().

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).map_or_else(lambda: 0, lambda x: x * 10)
        20
        >>> pc.NONE.map_or_else(lambda: 0, lambda x: x * 10)
        0

        ```
        """
        return f(self.unwrap()) if self.is_some() else default()


@dataclass(slots=True)
class Some[T](Option[T]):
    """
    Option variant representing the presence of a value.

    Args:
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
        """Returns `True` for `Some`."""
        return True

    def is_none(self) -> TypeIs[_None]:  # type: ignore[misc]
        """Returns `False` for `Some`."""
        return False

    def unwrap(self) -> T:
        """
        Returns the contained value.

        Returns:
            T: The contained value.
        """
        return self.value


@dataclass(slots=True)
class _None(Option[Any]):
    """
    Option variant representing the absence of a value.

    Example:
    ```python
    >>> import pyochain as pc
    >>> pc.NONE
    NONE

    ```
    """

    def __repr__(self) -> str:
        return "NONE"

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
