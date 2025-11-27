from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from .._core import Pipeable

if TYPE_CHECKING:
    from ._result import Result


class OptionUnwrapError(RuntimeError): ...


class Option[T](Pipeable, ABC):
    """Type `Option[T]` represents an optional value.

    Every Option is either:

    - `Some` and contains a value
    - `None`, and does not.

    This is a common type in Rust, and is used to represent values that may be absent.

    In python, this is best tought of a an union type `T | None`,
    but with additional methods to operate on the contained value in a functional style.

    `Option[T]` and/or `T | None` types are very useful, as they have a number of uses:

    - Initial values
    - Union types
    - Return value where None is returned on error
    - Optional class fields
    - Optional function arguments

    The fact that `T | None` is a very common pattern in python,
    but without a dedicated structure/handling, leads to:

    - a lot of boilerplate code
    - potential bugs (even with type checkers)
    - less readable code (where does the None come from? is it expected?).

    `Option[T]` instances are commonly paired with pattern matching.
    This allow to query the presence of a value and take action, always accounting for the None case.
    """

    @staticmethod
    def from_[V](value: V | None) -> Option[V]:
        """Creates an `Option[V]` from a value that may be `None`.

        Args:
            value (V | None): The value to convert into an `Option[V]`.

        Returns:
            Option[V]: `Some(value)` if the value is not `None`, otherwise `NONE`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Option.from_(42)
        Some(value=42)
        >>> pc.Option.from_(None)
        NONE

        ```
        """
        from ._states import NONE, Some

        return Some(value) if value is not None else NONE

    @abstractmethod
    def is_some(self) -> bool:
        """Returns `True` if the option is a `Some` value.

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
    def is_none(self) -> bool:
        """Returns `True` if the option is a `None` value.

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
        """Returns the contained `Some` value.

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
        """Returns the contained `Some` value.

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
        """Returns the contained `Some` value or a provided default.

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
        """Returns the contained `Some` value or computes it from a function.

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
        """Maps an `Option[T]` to `Option[U]`.

        Done by applying a function to a contained `Some` value,
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
        from ._states import NONE, Some

        return Some(f(self.unwrap())) if self.is_some() else NONE

    def and_then[U](self, f: Callable[[T], Option[U]]) -> Option[U]:
        """Calls a function if the option is `Some`, otherwise returns `None`.

        Args:
            f (Callable[[T], Option[U]]): The function to call with the `Some` value.

        Returns:
            Option[U]: The result of the function if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def sq(x: int) -> Option[int]:
        ...     return pc.Some(x * x)
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
        from ._states import NONE

        return f(self.unwrap()) if self.is_some() else NONE

    def or_else(self, f: Callable[[], Option[T]]) -> Option[T]:
        """Returns the `Option[T]` if it contains a value, otherwise calls a function and returns the result.

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
        ...     return pc.Some("vikings")
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
        """Returns an `Option[tuple[T, U]]` containing a tuple of the values if both options are `Some`, otherwise returns `NONE`.

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
        from ._states import NONE, Some

        if self.is_some() and other.is_some():
            return Some((self.unwrap(), other.unwrap()))
        return NONE

    def zip_with[U, R](self, other: Option[U], f: Callable[[T, U], R]) -> Option[R]:
        """Zips `self` and another `Option` with function `f`.

        If `self` is `Some(s)` and other is `Some(o)`, this method returns `Some(f(s, o))`.

        Otherwise, `NONE` is returned.

        Args:
            other (Option[U]): The second option.
            f (Callable[[T, U], R]): The function to apply to the unwrapped values.

        Returns:
            Option[R]: The resulting option after applying the function.

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
        from ._states import NONE, Some

        if self.is_some() and other.is_some():
            return Some(f(self.unwrap(), other.unwrap()))
        return NONE

    def ok_or[E](self, err: E) -> Result[T, E]:
        """Converts the option to a `Result`.

        Args:
            err (E): The error value to use if the option is `NONE`.

        Returns:
            Result[T, E]: `Ok(v)` if `Some(v)`, otherwise `Err(err)`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(1).ok_or('fail')
        Ok(value=1)
        >>> pc.NONE.ok_or('fail')
        Err(error='fail')

        ```
        """
        from ._states import Err, Ok

        return Ok(self.unwrap()) if self.is_some() else Err(err)

    def ok_or_else[E](self, err: Callable[[], E]) -> Result[T, E]:
        """Converts the option to a Result.

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
        from ._states import Err, Ok

        return Ok(self.unwrap()) if self.is_some() else Err(err())

    def map_or[U](self, f: Callable[[T], U], default: U) -> U:
        """Returns the result of applying a function to the contained value if Some, otherwise returns the default value.

        Args:
            f (Callable[[T], U]): The function to apply to the contained value.
            default (U): The default value to return if NONE.

        Returns:
            U: The result of f(self.unwrap()) if Some, otherwise default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).map_or(lambda x: x * 10, 0)
        20
        >>> pc.NONE.map_or(lambda x: x * 10, 0)
        0

        ```
        """
        return f(self.unwrap()) if self.is_some() else default

    def map_or_else[U](self, f: Callable[[T], U], default: Callable[[], U]) -> U:
        """Returns the result of applying a function to the contained value if Some, otherwise computes a default value.

        Args:
            f (Callable[[T], U]): The function to apply to the contained value.
            default (Callable[[], U]): A function returning the default value if NONE.

        Returns:
            U: The result of f(self.unwrap()) if Some, otherwise default().

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).map_or_else(lambda x: x * 10, lambda: 0)
        20
        >>> pc.NONE.map_or_else(lambda x: x * 10, lambda: 0)
        0

        ```
        """
        return f(self.unwrap()) if self.is_some() else default()

    def inspect(self, f: Callable[[T], object]) -> Option[T]:
        """Applies a function to the contained `Some` value, returning the original `Option`.

        This mirrors :meth:`Result.inspect`, allowing side effects
        (logging, debugging, metrics, etc.) on the wrapped value without changing it.

        Args:
            f (Callable[[T], object]): Function to apply to the `Some` value.

        Returns:
            Option[T]: The original option, unchanged.

        Example:
        ```python
        >>> import pyochain as pc
        >>> seen: list[int] = []
        >>> pc.Some(2).inspect(lambda x: seen.append(x))
        Some(value=2)
        >>> seen
        [2]
        >>> pc.NONE.inspect(lambda x: seen.append(x))
        NONE
        >>> seen
        [2]

        ```
        """
        if self.is_some():
            f(self.unwrap())
        return self
