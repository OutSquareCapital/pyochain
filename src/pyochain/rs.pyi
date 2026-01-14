from abc import ABC
from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Final, Protocol, Self, final, overload

from ._iter import Iter

# Mixin classes for pipeable and checkable methods

class Pipeable(Protocol):
    """Mixin class providing pipeable methods for fluent chaining."""

    def into[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Convert `Self` to `R`.

        This method allows to pipe the instance into an object or function that can convert `Self` into another type.

        Conceptually, this allow to do `x.into(f)` instead of `f(x)`, hence keeping a fluent chaining style.

        Args:
            func (Callable[Concatenate[Self, P], R]): Function for conversion.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            R: The converted value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> from collections.abc import Sequence
        >>> import hashlib
        >>> def sha256_hex(data: Sequence[int]) -> str:
        ...     return hashlib.sha256(bytes(data)).hexdigest()
        >>>
        >>> pc.Seq([1, 2, 3]).into(sha256_hex)
        '039058c6f2c0cb492c533b0a4d14ef77cc0f78abccced5287d84a1a2011cfb81'

        ```
        """

    def inspect[**P](
        self,
        func: Callable[Concatenate[Self, P], object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Pass `Self` to **func** to perform side effects without altering the data.

        This method is very useful for debugging or passing the instance to other functions for side effects, without breaking the fluent method chaining.

        Args:
            func (Callable[Concatenate[Self, P], object]): Function to apply to the instance for side effects.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Self: The instance itself, unchanged.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 4]).inspect(print).last()
        Seq(1, 2, 3, 4)
        4

        ```
        """

class Checkable(Protocol):
    """Mixin class providing conditional chaining methods based on truthiness.

    This class provides methods inspired by Rust's `bool` type for conditional
    execution and wrapping in `Option` or `Result` types.

    All methods evaluate the instance's truthiness to determine their behavior.
    """

    def then[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Option[R]:
        """Wrap `Self` in an `Option[R]` based on its truthiness.

        `R` being the return type of **func**.

        The function is only called if `Self` evaluates to `True` (lazy evaluation).

        Truthiness is determined by `__bool__()` if defined, otherwise by `__len__()` if defined (returning `False` if length is 0), otherwise all instances are truthy (Python's default behavior).

        Args:
            func (Callable[Concatenate[Self, P], R]): A callable that returns the value to wrap in Some.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Option[R]: `Some(R)` if self is truthy, `NONE` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).then(lambda s: s.sum())
        Some(6)
        >>> pc.Seq([]).then(lambda s: s.sum())
        NONE

        ```
        """

    def then_some(self) -> Option[Self]:
        """Wraps `Self` in an `Option[Self]` based on its truthiness.

        Truthiness is determined by `__bool__()` if defined, otherwise by `__len__()` if defined (returning `False` if length is 0), otherwise all instances are truthy (Python's default behavior).

        Returns:
            Option[Self]: `Some(self)` if self is truthy, `NONE` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).then_some()
        Some(Seq(1, 2, 3))
        >>> pc.Seq([]).then_some()
        NONE

        ```
        """
    def ok_or[E](self, err: E) -> Result[Self, E]:
        """Wrap `Self` in a `Result[Self, E]` based on its truthiness.

        Truthiness is determined by `__bool__()` if defined, otherwise by `__len__()` if defined (returning `False` if length is 0), otherwise all instances are truthy (Python's default behavior).

        Args:
            err (E): The error value to wrap in Err if self is falsy.

        Returns:
            Result[Self, E]: `Ok(self)` if self is truthy, `Err(err)` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).ok_or("empty")
        Ok(Seq(1, 2, 3))
        >>> pc.Seq([]).ok_or("empty")
        Err('empty')

        ```
        """

    def ok_or_else[**P, E](
        self,
        func: Callable[Concatenate[Self, P], E],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[Self, E]:
        """Wrap `Self` in a `Result[Self, E]` based on its truthiness.

        `E` being the return type of **func**.

        The function is only called if self evaluates to False.

        Truthiness is determined by `__bool__()` if defined, otherwise by `__len__()` if defined (returning `False` if length is 0), otherwise all instances are truthy (Python's default behavior).

        Args:
            func (Callable[Concatenate[Self, P], E]): A callable that returns the error value to wrap in Err.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Result[Self, E]: Ok(self) if self is truthy, Err(f(...)) otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).ok_or_else(lambda s: f"empty seq")
        Ok(Seq(1, 2, 3))
        >>> pc.Seq([]).ok_or_else(lambda s: f"empty seq")
        Err('empty seq')

        ```
        """

# Option types
class OptionUnwrapError(RuntimeError): ...

class Option[T](Pipeable):
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

    def __new__[V](cls, value: V | None) -> Option[V]:
        """Creates an `Option[V]` from a value that may be `None`.

        When calling `Option(value)`, this method automatically redirects to:
        - `Some(value)` if the value is not `None`
        - `NONE` if the value is `None`

        Args:
            value (V | None): The value to convert into an `Option[V]`.

        Returns:
            Option[V]: `Some(value)` if the value is not `None`, otherwise `NONE`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Option(42)
        Some(42)
        >>> pc.Option(None)
        NONE

        ```
        """

    def __bool__(self) -> None:
        """Prevent implicit `Some|None` value checking in boolean contexts.

        Raises:
            TypeError: Always, to prevent implicit `Some|None` value checking.

        Example:
        ```python
        >>> import pyochain as pc
        >>> x = pc.Some(42)
        >>> bool(x)
        Traceback (most recent call last):
        ...
        TypeError: Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.

        ```
        """

    @staticmethod
    def if_true[V](value: V, *, predicate: Callable[[V], bool]) -> Option[V]:
        """Creates an `Option[V]` based on a **predicate** condition on the provided **value**.

        Args:
            value (V): The value to wrap in `Some` if the condition is `True`.
            predicate (Callable[[V], bool]): The condition to evaluate.

        Returns:
            Option[V]: `Some(value)` if the condition is `True`, otherwise `NONE`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Option.if_true(42, predicate=lambda x: x == 42)
        Some(42)
        >>> pc.Option.if_true(21, predicate=lambda x: x == 42)
        NONE
        >>> from pathlib import Path
        >>> pc.Option.if_true(Path("README.md"), predicate=Path.exists).map(str)
        Some('README.md')
        >>> pc.Option.if_true(Path("README.md"), predicate=lambda p: p.exists()).map(str) # Same as above
        Some('README.md')

        ```
        """

    @staticmethod
    def if_some[V](value: V) -> Option[V]:
        """Creates an `Option[V]` based on the truthiness of a value.

        Args:
            value (V): The value to evaluate.

        Returns:
            Option[V]: `Some(value)` if the value is truthy, otherwise `NONE`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Option.if_some(42)
        Some(42)
        >>> pc.Option.if_some(0)
        NONE
        >>> pc.Option.if_some("hello")
        Some('hello')
        >>> pc.Option.if_some("")
        NONE

        ```
        """
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

    @overload
    def map_star[R](
        self: Option[tuple[Any]],
        func: Callable[[Any], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, R](
        self: Option[tuple[T1, T2]],
        func: Callable[[T1, T2], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, R](
        self: Option[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, R](
        self: Option[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, R](
        self: Option[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
    ) -> Option[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
    ) -> Option[R]: ...
    def map_star[U: Iterable[Any], R](
        self: Option[U],
        func: Callable[..., R],
    ) -> Option[R]:
        """Maps an `Option[Iterable]` to `Option[U]` by unpacking the iterable into the function.

        Done by applying a function to a contained `Some` value,
        leaving a `None` value untouched.

        Args:
            func (Callable[..., R]): The function to apply to the unpacked `Some` value.

        Returns:
            Option[R]: A new `Option` with the mapped value if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some((2, 3)).map_star(lambda x, y: x + y)
        Some(5)
        >>> pc.NONE.map_star(lambda x, y: x + y)
        NONE

        ```
        """

    @overload
    def and_then_star[R](
        self: Option[tuple[Any]],
        func: Callable[[Any], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, R](
        self: Option[tuple[T1, T2]],
        func: Callable[[T1, T2], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, R](
        self: Option[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, R](
        self: Option[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, R](
        self: Option[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], Option[R]],
    ) -> Option[R]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Option[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], Option[R]],
    ) -> Option[R]: ...
    def and_then_star[U: Iterable[Any], R](
        self: Option[U],
        func: Callable[..., Option[R]],
    ) -> Option[R]:
        """Calls a function if the option is `Some`, unpacking the iterable into the function.

        Args:
            func (Callable[..., Option[R]]): The function to call with the unpacked `Some` value.

        Returns:
            Option[R]: The result of the function if `Some`, otherwise `None`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some((2, 3)).and_then_star(lambda x, y: pc.Some(x + y))
        Some(5)
        >>> pc.NONE.and_then_star(lambda x, y: pc.Some(x + y))
        NONE

        ```
        """

    def ne(self, other: Option[T]) -> bool:
        """Checks if two `Option[T]` instances are not equal.

        Args:
            other (Option[T]): The other `Option[T]` instance to compare with.

        Returns:
            bool: `True` if both instances are not equal, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(42).ne(pc.Some(21))
        True
        >>> pc.Some(42).ne(pc.Some(42))
        False
        >>> pc.Some(42).ne(pc.NONE)
        True
        >>> pc.NONE.ne(pc.NONE)
        False

        ```
        """

    # Abstract methods ----------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Checks if this `Option` and *other* are equal.

        A plain Python `None` is considered equal to a `pyochain.NoneOption` instance.

        A plain value of type `T` is considered equal to a `pyochain.Some[T]` instance.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: `True` if both instances are equal, `False` otherwise.

        See Also:
            - `Option.eq` for a type-safe, performant version that only accepts `Option[T]` instances.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(42) == pc.Some(42)
        True
        >>> pc.Some(42) == pc.Some(21)
        False
        >>> pc.Some(42) == pc.NONE
        False
        >>> pc.NONE == pc.NONE
        True
        >>> pc.NONE == None
        True
        >>> pc.Some(42) == 42
        True

        ```
        """

    def eq(self, other: Option[T]) -> bool:
        """Checks if two `Option[T]` instances are equal.

        Note:
            This method behave similarly to `__eq__`, but only accepts `Option[T]` instances as argument.

            This avoids runtime isinstance checks (we check for boolean `is_some()`, which is a simple function call), and is more type-safe.

        Args:
            other (Option[T]): The other `Option[T]` instance to compare with.

        Returns:
            bool: `True` if both instances are equal, `False` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Some(42).eq(pc.Some(42))
        True
        >>> pc.Some(42).eq(pc.Some(21))
        False
        >>> pc.Some(42).eq(pc.NONE)
        False
        >>> pc.NONE.eq(pc.NONE)
        True

        ```
        """

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

        Example:
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

        Example:
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
        OptionUnwrapError: called `unwrap` on a `None`

        ```
        """

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
        OptionUnwrapError: fruits are healthy (called `expect` on a `None`)

        ```
        """

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

    def or_(self, optb: Option[T]) -> Option[T]:
        """Returns the option if it contains a value, otherwise returns optb.

        Args:
            optb (Option[T]): The option to return if the original option is `NONE`.

        Returns:
            Option[T]: The original option if it is `Some`, otherwise `optb`.

        Example:
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


        Example:
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

    def zip_with[U, R](self, other: Option[U], f: Callable[[T, U], R]) -> Option[R]:
        """Zips `self` and another `Option` with function `f`.

        If `self` is `Some(s)` and other is `Some(o)`, this method returns `Some(f(s, o))`.

        Otherwise, `NONE` is returned.

        Args:
            other (Option[U]): The second option.
            f (Callable[[T, U], R]): The function to apply to the unwrapped values.

        Returns:
            Option[R]: The resulting option after applying the function.

        Example:
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
        >>> pc.NONE.zip_with(y, Point)
        NONE

        ```
        """

    def reduce(self, other: Option[T], func: Callable[[T, T], T]) -> Option[T]:
        """Reduces two options into one, using the provided function if both are Some.

        If **self** is `Some(s)` and **other** is `Some(o)`, this method returns `Some(func(s, o))`.

        Otherwise, if only one of **self** and **other** is `Some`, that value is returned.

        If both **self** and **other** are `NONE`, `NONE` is returned.

        Args:
            other (Option[T]): The second option.
            func (Callable[[T, T], T]): The function to apply to the unwrapped values.

        Returns:
            Option[T]: The resulting option after reduction.

        Example:
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

    def transpose[E](self: Option[Result[T, E]]) -> Result[Option[T], E]:
        """Transposes an `Option` of a `Result` into a `Result` of an `Option`.

        `Some(Ok[T])` is mapped to `Ok(Some[T])`, `Some(Err[E])` is mapped to `Err[E]`, and `NONE` will be mapped to `Ok(NONE)`.

        Returns:
            Result[Option[T], E]: The transposed result.

        Example:
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

    def xor(self, optb: Option[T]) -> Option[T]:
        """Returns `Some` if exactly one of **self**, optb is `Some`, otherwise returns `NONE`.

        Args:
            optb (Option[T]): The other option to compare with.

        Returns:
            Option[T]: `Some` value if exactly one option is `Some`, otherwise `NONE`.

        Example:
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

@final
class Some[T](Option[T]):
    """Option variant representing the presence of a value.

    For more documentation, see the `Option[T]` class.

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
    __match_args__ = ("value",)

    def __new__[V](cls, value: V) -> Some[V]:
        """Bypass Option's redirect by directly creating a Some instance."""

@final
class NoneOption[T](Option[T]):
    """Option variant representing the absence of a value.

    This class is not supposed to be instanciated by the user.
    Instead, pyochain provide the `NONE` singleton (for performance reasons).

    For more documentation, see the `Option[T]` class.
    """

NONE: Final[NoneOption[Any]] = ...
"""Singleton instance representing the absence of a value.
This is the only instance of `NoneOption` that should be used, and is similar to the logic used by `None` in standard Python.
This allows you to use `is pc.NONE` checks for identity, and improves performance by avoiding unnecessary allocations and instanciations.
"""

class ResultUnwrapError(RuntimeError): ...

class Result[T, E](Pipeable, ABC):
    """`Result[T, E]` is the type used for returning and propagating errors.

    It is a class that can represent two variants, `Ok[T]`, representing success and containing a value, and `Err[E]`, representing error and containing an error value.

    Functions return `Result` whenever errors are expected and recoverable.

    For example, I/O or web requests can fail for many reasons, and using `Result` forces the caller to handle the possibility of failure.

    This is directly inspired by Rust's `Result` type, and provides similar functionality for error handling in Python.

    """

    __slots__ = ()

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

    def iter(self) -> Iter[T]:
        """Returns a `Iter[T]` over the possibly contained value.

        The iterator yields one value if the result is `Ok`, otherwise none.

        Returns:
            Iter[T]: An iterator over the `Ok` value, or empty if `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok(7).iter().next()
        Some(7)
        >>> pc.Err("nothing!").iter().next()
        NONE

        ```
        """

    @overload
    def map_star[R](
        self: Result[tuple[Any], E],
        func: Callable[[Any], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, R](
        self: Result[tuple[T1, T2], E],
        func: Callable[[T1, T2], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, R](
        self: Result[tuple[T1, T2, T3], E],
        func: Callable[[T1, T2, T3], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, R](
        self: Result[tuple[T1, T2, T3, T4], E],
        func: Callable[[T1, T2, T3, T4], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, R](
        self: Result[tuple[T1, T2, T3, T4, T5], E],
        func: Callable[[T1, T2, T3, T4, T5], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6], E],
        func: Callable[[T1, T2, T3, T4, T5, T6], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
    ) -> Result[R, E]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
    ) -> Result[R, E]: ...
    def map_star[U: Iterable[Any], R](
        self: Result[U, E],
        func: Callable[..., R],
    ) -> Result[R, E]:
        """Maps a `Result[tuple, E]` to `Result[R, E]` by unpacking the tuple.

        Done by applying a function to a contained `Ok` value (which is expected to be a tuple).

        Args:
            func (Callable[..., R]): The function to apply to the unpacked `Ok` value.

        Returns:
            Result[R, E]: A new `Result` with the mapped value if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Ok((2, 3)).map_star(lambda x, y: x + y)
        Ok(5)
        >>> pc.Err("error").map_star(lambda x, y: x + y)
        Err('error')

        ```
        """

    @overload
    def and_then_star[R](
        self: Result[tuple[Any], E],
        func: Callable[[Any], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, R](
        self: Result[tuple[T1, T2], E],
        func: Callable[[T1, T2], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, R](
        self: Result[tuple[T1, T2, T3], E],
        func: Callable[[T1, T2, T3], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, R](
        self: Result[tuple[T1, T2, T3, T4], E],
        func: Callable[[T1, T2, T3, T4], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, R](
        self: Result[tuple[T1, T2, T3, T4, T5], E],
        func: Callable[[T1, T2, T3, T4, T5], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6], E],
        func: Callable[[T1, T2, T3, T4, T5, T6], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], Result[R, E]],
    ) -> Result[R, E]: ...
    @overload
    def and_then_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], E],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], Result[R, E]],
    ) -> Result[R, E]: ...
    def and_then_star[U: Iterable[Any], R](
        self: Result[U, E],
        func: Callable[..., Result[R, E]],
    ) -> Result[R, E]:
        """Calls a function if the result is `Ok`, unpacking the tuple.

        Done by applying a function to a contained `Ok` value (which is expected to be a tuple).

        Args:
            func (Callable[..., Result[R, E]]): The function to call with the unpacked `Ok` value.

        Returns:
            Result[R, E]: The result of the function if `Ok`, otherwise the original `Err`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def to_str(x: int, y: int) -> pc.Result[str, str]:
        ...     return pc.Ok(f"{x},{y}")
        >>> pc.Ok((2, 3)).and_then_star(to_str)
        Ok('2,3')
        >>> pc.Err("error").and_then_star(to_str)
        Err('error')


        ```
        """

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
        ResultUnwrapError: called `unwrap` on an `Err`: 'emergency failure'

        ```
        """

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
        ResultUnwrapError: called `unwrap_err` on Ok

        ```
        """

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
        ResultUnwrapError: Testing expect: 'emergency failure'

        ```
        """

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
        ResultUnwrapError: Testing expect_err: expected Err, got Ok(10)

        ```
        """

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

    def or_[F](self, res: Result[T, F]) -> Result[T, F]:
        """Returns res if the result is `Err`, otherwise returns the `Ok` value of **self**.

        Args:
            res (Result[T, F]): The result to return if the original result is `Err`.

        Returns:
            Result[T, F]: The original `Ok` value, or `res` if the original result is `Err`.

        Example:
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

@final
class Ok[T, E](Result[T, E]):
    """Represents a successful value.

    For more documentation, see the `Result[T, E]` class.

    Attributes:
        value (T): The contained successful value.
    """

    __match_args__ = ("value",)

    value: T
    def __new__(cls, value: T) -> Ok[T, Any]: ...

@final
class Err[T, E](Result[T, E]):
    """Represents an error value.

    For more documentation, see the `Result[T, E]` class.

    Attributes:
        error (E): The contained error value.
    """

    __match_args__ = ("error",)

    error: E
    def __new__(cls, error: E) -> Err[Any, E]: ...
