from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Concatenate, Never, cast

from ._core import Pipeable

if TYPE_CHECKING:
    from ._iter import Iter
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

    def __bool__(self) -> None:
        """Prevent implicit `Some|None` value checking in boolean contexts.

        Raises:
            TypeError: Always, to prevent implicit `Some|None` value checking.
        """
        msg = "Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead."
        raise TypeError(msg)

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
        Some(42)
        >>> pc.Option.from_(None)
        NONE

        ```
        """
        return cast(Option[V], Some(value) if value is not None else NONE)

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

    def is_some_and[**P](
        self,
        predicate: Callable[Concatenate[T, P], bool],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> bool:
        """Returns true if the option is a Some and the value inside of it matches a predicate.

        Args:
            predicate (Callable[Concatenate[T, P], bool]): The predicate to apply to the contained value.
            *args (P.args): Additional positional arguments to pass to predicate.
            **kwargs (P.kwargs): Additional keyword arguments to pass to predicate.

        Returns:
            bool: `True` if the option is `Some` and the predicate returns `True` for the contained value, `False` otherwise.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> x = pc.Some(2)
        >>> x.is_some_and(lambda x: x > 1)
        True

        >>> x = pc.Some(0)
        >>> x.is_some_and(lambda x: x > 1)
        False
        >>> x = pc.NONE
        >>> x.is_some_and(lambda x: x > 1)
        False
        >>> x = pc.Some("hello")
        >>> x.is_some_and(lambda x: len(x) > 1)
        True

        ```
        """
        return self.is_some() and predicate(self.unwrap(), *args, **kwargs)

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

    def is_none_or[**P](
        self, func: Callable[Concatenate[T, P], bool], *args: P.args, **kwargs: P.kwargs
    ) -> bool:
        """Returns true if the option is a None or the value inside of it matches a predicate.

        Args:
            func (Callable[Concatenate[T, P], bool]): The predicate to apply to the contained value.
            *args (P.args): Additional positional arguments to pass to func.
            **kwargs (P.kwargs): Additional keyword arguments to pass to func.

        Returns:
            bool: `True` if the option is `None` or the predicate returns `True` for the contained value, `False` otherwise.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).is_none_or(lambda x: x > 1)
        True
        >>> pc.Some(0).is_none_or(lambda x: x > 1)
        False
        >>> pc.NONE.is_none_or(lambda x: x > 1)
        True
        >>> pc.Some("hello").is_none_or(lambda x: len(x) > 1)
        True

        ```
        """
        return self.is_none() or func(self.unwrap(), *args, **kwargs)

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
        pyochain._option.OptionUnwrapError: called `unwrap` on a `None`

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
        pyochain._option.OptionUnwrapError: fruits are healthy (called `expect` on a `None`)

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

    def map[**P, R](
        self, f: Callable[Concatenate[T, P], R], *args: P.args, **kwargs: P.kwargs
    ) -> Option[R]:
        """Maps an `Option[T]` to `Option[U]`.

        Done by applying a function to a contained `Some` value,
        leaving a `None` value untouched.

        Args:
            f (Callable[Concatenate[T, P], R]): The function to apply to the `Some` value.
            *args (P.args): Additional positional arguments to pass to f.
            **kwargs (P.kwargs): Additional keyword arguments to pass to f.

        Returns:
            Option[R]: A new `Option` with the mapped value if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some("Hello, World!").map(len)
        Some(13)
        >>> pc.NONE.map(len)
        NONE

        ```
        """
        return Some(f(self.unwrap(), *args, **kwargs)) if self.is_some() else NONE

    def flatten[U](self: Option[Option[U]]) -> Option[U]:
        """Flattens a nested `Option`.

        Converts an `Option[Option[U]]` into an `Option[U]` by removing one level of nesting.

        Equivalent to `Option.and_then(lambda x: x)`.

        Returns:
            Option[U]: The flattened option.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(pc.Some(42)).flatten()
        Some(42)
        >>> pc.Some(pc.NONE).flatten()
        NONE
        >>> pc.NONE.flatten()
        NONE

        ```
        """
        return self.and_then(lambda x: x)

    def and_[U](self, optb: Option[U]) -> Option[U]:
        """Returns `NONE` if the option is `NONE`, otherwise returns optb.

        This is similar to `and_then`, except that the value is passed directly instead of through a closure.

        Args:
            optb (Option[U]): The option to return if the original option is `NONE`
        Returns:
            Option[U]: `NONE` if the original option is `NONE`, otherwise `optb`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).and_(pc.NONE)
        NONE
        >>> pc.NONE.and_(pc.Some("foo"))
        NONE
        >>> pc.Some(2).and_(pc.Some("foo"))
        Some('foo')
        >>> pc.NONE.and_(pc.NONE)
        NONE

        ```
        """
        return optb if self.is_some() else NONE

    def or_(self, optb: Option[T]) -> Option[T]:
        """Returns the option if it contains a value, otherwise returns optb.

        Args:
            optb (Option[T]): The option to return if the original option is `NONE`.

        Returns:
            Option[T]: The original option if it is `Some`, otherwise `optb`.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).or_(pc.NONE)
        Some(2)
        >>> pc.NONE.or_(pc.Some(100))
        Some(100)
        >>> pc.Some(2).or_(pc.Some(100))
        Some(2)
        >>> pc.NONE.or_(pc.NONE)
        NONE

        ```
        """
        return self if self.is_some() else optb

    def and_then[**P, R](
        self,
        f: Callable[Concatenate[T, P], Option[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Option[R]:
        """Calls a function if the option is `Some`, otherwise returns `None`.

        Args:
            f (Callable[Concatenate[T, P], Option[R]]): The function to call with the `Some` value.
            *args (P.args): Additional positional arguments to pass to f.
            **kwargs (P.kwargs): Additional keyword arguments to pass to f.

        Returns:
            Option[R]: The result of the function if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def sq(x: int) -> Option[int]:
        ...     return pc.Some(x * x)
        >>> def nope(x: int) -> Option[int]:
        ...     return pc.NONE
        >>> pc.Some(2).and_then(sq).and_then(sq)
        Some(16)
        >>> pc.Some(2).and_then(sq).and_then(nope)
        NONE
        >>> pc.Some(2).and_then(nope).and_then(sq)
        NONE
        >>> pc.NONE.and_then(sq).and_then(sq)
        NONE

        ```
        """
        return f(self.unwrap(), *args, **kwargs) if self.is_some() else NONE

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
        Some('barbarians')
        >>> pc.NONE.or_else(vikings)
        Some('vikings')
        >>> pc.NONE.or_else(nobody)
        NONE

        ```
        """
        return self if self.is_some() else f()

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
        Ok(1)
        >>> pc.NONE.ok_or('fail')
        Err('fail')

        ```
        """
        from ._result import Err, Ok

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
        Ok(1)
        >>> pc.NONE.ok_or_else(lambda: 'fail')
        Err('fail')

        ```
        """
        from ._result import Err, Ok

        return Ok(self.unwrap()) if self.is_some() else Err(err())

    def map_or[**P, R](
        self,
        default: R,
        f: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Returns the result of applying a function to the contained value if Some, otherwise returns the default value.

        Args:
            default (R): The default value to return if NONE.
            f (Callable[Concatenate[T, P], R]): The function to apply to the contained value.
            *args (P.args): Additional positional arguments to pass to f.
            **kwargs (P.kwargs): Additional keyword arguments to pass to f.

        Returns:
            R: The result of f(self.unwrap()) if Some, otherwise default.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).map_or(0, lambda x: x * 10)
        20
        >>> pc.NONE.map_or(0, lambda x: x * 10)
        0

        ```
        """
        return f(self.unwrap(), *args, **kwargs) if self.is_some() else default

    def map_or_else[**P, R](self, default: Callable[[], R], f: Callable[[T], R]) -> R:
        """Returns the result of applying a function to the contained value if Some, otherwise computes a default value.

        Args:
            default (Callable[[], R]): A function returning the default value if NONE.
            f (Callable[[T], R]): The function to apply to the contained value.

        Returns:
            R: The result of f(self.unwrap()) if Some, otherwise default().

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

    def filter[**P, R](
        self,
        predicate: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Option[T]:
        """Returns None if the option is None, otherwise calls predicate with the wrapped value.

        This function works similar to `Iter.filter` in the sense that we only keep the value if it matches a predicate.

        You can imagine the `Option[T]` being an iterator over one or zero elements.

        Args:
            predicate (Callable[Concatenate[T, P], R]): The predicate to apply to the contained value.
            *args (P.args): Additional positional arguments to pass to predicate.
            **kwargs (P.kwargs): Additional keyword arguments to pass to predicate.

        Returns:
            Option[T]: `Some[T]` if predicate returns true (where T is the wrapped value), `NONE` if predicate returns false.


        Examples:
        ```python
        >>> import pyochain as pc
        >>>
        >>> def is_even(n: int) -> bool:
        ...     return n % 2 == 0
        >>>
        >>> pc.NONE.filter(is_even)
        NONE
        >>> pc.Some(3).filter(is_even)
        NONE
        >>> pc.Some(4).filter(is_even)
        Some(4)

        ```
        """
        return (
            self
            if self.is_some() and predicate(self.unwrap(), *args, **kwargs)
            else NONE
        )

    def iter(self) -> Iter[T]:
        """Creates an `Iter` over the optional value.

        - If the option is `Some(value)`, the iterator yields `value`.
        - If the option is `NONE`, the iterator yields nothing.

        Equivalent to `Iter((self,))`.

        Returns:
            Iter[T]: An iterator over the optional value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(42).iter().next()
        Some(42)
        >>> pc.NONE.iter().next()
        NONE

        ```
        """
        from ._iter import Iter

        return Iter((self.unwrap(),)) if self.is_some() else Iter(())

    def inspect[**P](
        self, f: Callable[Concatenate[T, P], object], *args: P.args, **kwargs: P.kwargs
    ) -> Option[T]:
        """Applies a function to the contained `Some` value, returning the original `Option`.

        This allows side effects (logging, debugging, metrics, etc.) on the wrapped value without changing it.

        Args:
            f (Callable[Concatenate[T, P], object]): Function to apply to the `Some` value.
            *args (P.args): Additional positional arguments to pass to f.
            **kwargs (P.kwargs): Additional keyword arguments to pass to f.

        Returns:
            Option[T]: The original option, unchanged.

        Example:
        ```python
        >>> import pyochain as pc
        >>> seen: list[int] = []
        >>> pc.Some(2).inspect(lambda x: seen.append(x))
        Some(2)
        >>> seen
        [2]
        >>> pc.NONE.inspect(lambda x: seen.append(x))
        NONE
        >>> seen
        [2]

        ```
        """
        if self.is_some():
            f(self.unwrap(), *args, **kwargs)
        return self

    def unzip[U](self: Option[tuple[T, U]]) -> tuple[Option[T], Option[U]]:
        """Unzips an `Option` of a tuple into a tuple of `Option`s.

        If the option is `Some((a, b))`, this method returns `(Some(a), Some(b))`.
        If the option is `NONE`, it returns `(NONE, NONE)`.

        Returns:
            tuple[Option[T], Option[U]]: A tuple containing two options.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some((1, 'a')).unzip()
        (Some(1), Some('a'))
        >>> pc.NONE.unzip()
        (NONE, NONE)

        ```
        """
        if self.is_some():
            a, b = self.unwrap()
            return Some(a), Some(b)
        return NONE, NONE

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
        Some((1, 'a'))
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
        Some(Point(x=17.5, y=42.7))
        >>> x.zip_with(pc.NONE, Point)
        NONE

        ```
        """
        if self.is_some() and other.is_some():
            return Some(f(self.unwrap(), other.unwrap()))
        return NONE

    def reduce[U](self, other: Option[T], func: Callable[[T, T], T]) -> Option[T]:
        """Reduces two options into one, using the provided function if both are Some.

        If **self** is `Some(s)` and **other** is `Some(o)`, this method returns `Some(func(s, o))`.

        Otherwise, if only one of **self** and **other** is `Some`, that value is returned.

        If both **self** and **other** are `NONE`, `NONE` is returned.

        Args:
            other (Option[T]): The second option.
            func (Callable[[T, T], T]): The function to apply to the unwrapped values.

        Returns:
            Option[T]: The resulting option after reduction.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> s12 = pc.Some(12)
        >>> s17 = pc.Some(17)
        >>>
        >>> def add(a: int, b: int) -> int:
        ...     return a + b
        >>>
        >>> s12.reduce(s17, add)
        Some(29)
        >>> s12.reduce(pc.NONE, add)
        Some(12)
        >>> pc.NONE.reduce(s17, add)
        Some(17)
        >>> pc.NONE.reduce(pc.NONE, add)
        NONE

        ```
        """
        if self.is_some() and other.is_some():
            return Some(func(self.unwrap(), other.unwrap()))
        if self.is_some():
            return self
        if other.is_some():
            return other
        return NONE

    def transpose[E](self: Option[Result[T, E]]) -> Result[Option[T], E]:
        """Transposes an `Option` of a `Result` into a `Result` of an `Option`.

        `Some(Ok[T])` is mapped to `Ok(Some[T])`, `Some(Err[E])` is mapped to `Err[E]`, and `NONE` will be mapped to `Ok(NONE)`.

        Returns:
            Result[Option[T], E]: The transposed result.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(pc.Ok(5)).transpose()
        Ok(Some(5))
        >>> pc.NONE.transpose()
        Ok(NONE)
        >>> pc.Some(pc.Err("error")).transpose()
        Err('error')

        ```
        """
        from ._result import Err, Ok

        if self.is_some():
            inner = self.unwrap()
            if inner.is_ok():
                return Ok(Option.from_(inner.unwrap()))
            return Err(inner.unwrap_err())
        return Ok(Option.from_(None))

    def xor(self, optb: Option[T]) -> Option[T]:
        """Returns `Some` if exactly one of **self**, optb is `Some`, otherwise returns `NONE`.

        Args:
            optb (Option[T]): The other option to compare with.

        Returns:
            Option[T]: `Some` value if exactly one option is `Some`, otherwise `NONE`.

        Examples:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(2).xor(pc.NONE)
        Some(2)
        >>> pc.NONE.xor(pc.Some(2))
        Some(2)
        >>> pc.Some(2).xor(pc.Some(2))
        NONE
        >>> pc.NONE.xor(pc.NONE)
        NONE

        ```
        """
        if self.is_some() and not optb.is_some():
            return self
        if not self.is_some() and optb.is_some():
            return optb
        return NONE


@dataclass(slots=True)
class Some[T](Option[T]):
    """Option variant representing the presence of a value.

    Attributes:
        value (T): The contained value.

    Example:
    ```python
    >>> import pyochain as pc
    >>> pc.Some(42)
    Some(42)

    ```

    """

    value: T

    def __repr__(self) -> str:
        return f"Some({self.value!r})"

    def is_some(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def unwrap(self) -> T:
        return self.value


@dataclass(slots=True)
class NoneOption[T](Option[T]):
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


NONE: NoneOption[Any] = NoneOption()
"""Singleton instance representing the absence of a value."""
