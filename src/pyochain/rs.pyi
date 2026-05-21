from collections.abc import Callable, Iterable
from typing import (
    Any,
    Concatenate,
    Final,
    Protocol,
    Self,
    final,
    overload,
    override,
    type_check_only,
)

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
            >>> from pyochain import Seq, Result, Ok, Err
            >>> from collections.abc import Sequence
            >>> def check_data(data: Sequence[int]) -> Result[Sequence[int], str]:
            ...     if len(data) == 0:
            ...         return Err("Empty data")
            ...     return Ok(data)
            >>>
            >>> def handle_result(res: Result[Sequence[int], str]) -> str:
            ...     match res:
            ...         case Ok(data):
            ...             return f"Data is valid: {data}"
            ...         case Err(err):
            ...             return f"Data is invalid: {err}"
            >>>
            >>> Seq((1, 2, 3)).into(check_data).into(handle_result)
            'Data is valid: Seq(1, 2, 3)'

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
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3, 4)).inspect(print).last()
            Seq(1, 2, 3, 4)
            4

            ```
        """

class Checkable(Protocol):
    """Mixin class providing conditional chaining methods based on truthiness.

    This class provides methods inspired by Rust's `bool` type for conditional
    execution and wrapping in `Option` or `Result` types.

    All methods evaluate the instance's truthiness to determine their behavior.

    Truthiness being determined by:

    - `__bool__()` if defined
    - otherwise by `__len__()` if defined (returning `False` if length is 0)
    - otherwise all instances are truthy (Python's default behavior).

    This can be very handy to cover the common pattern of checking if a collection is empty or not.

    You can then explicitly handle each situation with `Option` or `Result` types, without breaking the fluent method chaining.

    Tip:
        This class is compiled in Rust with Pyo3 bindings.

        This means that even pure Python classes inheriting from `Checkable` can call these methods with builtin-like performance.

    Example:
        Pyochain collections can efficiently check for emptiness and execute code conditionally natively.
        ```python
        >>> from pyochain import Seq
        >>> Seq((1, 2, 3)).then(sum)
        Some(6)
        >>> Seq(()).then(sum)
        NONE

        ```
        This can also be extended to any type, not just collections.
        ```python
        >>> from pyochain.abc import Checkable
        >>> class MyString(str, Checkable):
        ...     pass
        >>> MyString("hello").then(lambda s: s.upper())
        Some('HELLO')
        >>> MyString("").then(lambda s: s.upper())
        NONE

        ```
        This means that you can handle complex business logic in the same way.
        ```python
        >>> from dataclasses import dataclass
        >>> @dataclass(slots=True)
        ... class User(Checkable):
        ...     name: str
        ...     is_active: bool
        ...     age: int
        ...     def __bool__(self) -> bool:
        ...         return self.is_active and self.age >= 18
        ...
        ...     def describe(self) -> str:
        ...         return f"{self.name} is an active adult"
        >>>
        >>> alice = User("Alice", is_active=True, age=30).then(User.describe)
        >>> bob = (
        ...     User("Bob", is_active=False, age=24)
        ...     .then(User.describe)
        ...     .ok_or("Expected an active adult user")
        ...     .map_err(ValueError)
        ... )
        >>> alice
        Some('Alice is an active adult')
        >>> bob
        Err(ValueError('Expected an active adult user'))

        ```
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

        Args:
            func (Callable[Concatenate[Self, P], R]): A callable that returns the value to wrap in Some.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Option[R]: `Some(R)` if self is truthy, `NONE` otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).then(lambda s: s.sum())
            Some(6)
            >>> Seq(()).then(lambda s: s.sum())
            NONE

            ```
        """

    def then_some(self) -> Option[Self]:
        """Wraps `Self` in an `Option[Self]` based on its truthiness.

        Returns:
            Option[Self]: `Some(self)` if self is truthy, `NONE` otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).then_some()
            Some(Seq(1, 2, 3))
            >>> Seq(()).then_some()
            NONE

            ```
        """
    def ok_or[E](self, err: E) -> Result[Self, E]:
        """Wrap `Self` in a `Result[Self, E]` based on its truthiness.

        This method is the inverse of `err_or`.

        Args:
            err (E): The error value to wrap in Err if self is falsy.

        Returns:
            Result[Self, E]: `Ok(self)` if self is truthy, `Err(err)` otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).ok_or("empty")
            Ok(Seq(1, 2, 3))
            >>> Seq(()).ok_or("empty")
            Err('empty')

            ```
        """
    def err_or[T](self, ok: T) -> Result[T, Self]:
        """Wrap `Self` in a `Result[T, Self]` based on its truthiness.

        This method is the inverse of `ok_or`.

        Args:
            ok (T): The ok value to wrap in Ok if self is falsy.

        Returns:
            Result[T, Self]: `Ok(ok)` if self is truthy, `Err(self)` otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).err_or("should be empty")
            Err(Seq(1, 2, 3))
            >>> Seq(()).err_or("should be empty")
            Ok('should be empty')

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

        Args:
            func (Callable[Concatenate[Self, P], E]): A callable that returns the error value to wrap in Err.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Result[Self, E]: Ok(self) if self is truthy, Err(f(...)) otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).ok_or_else(lambda s: f"empty seq")
            Ok(Seq(1, 2, 3))
            >>> Seq(()).ok_or_else(lambda s: f"empty seq")
            Err('empty seq')

            ```
        """
    def err_or_else[**P, T](
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[T, Self]:
        """Wrap `Self` in a `Result[Self, E]` based on its truthiness.

        `E` being the return type of **func**.

        The function is only called if self evaluates to False.


        Args:
            func (Callable[Concatenate[Self, P], E]): A callable that returns the error value to wrap in Err.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Result[Self, E]: Ok(self) if self is truthy, Err(f(...)) otherwise.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).err_or_else(lambda s: "should be empty")
            Err(Seq(1, 2, 3))
            >>> Seq(()).err_or_else(lambda s: "should be empty")
            Ok('should be empty')

            ```
        """

# Option types
class OptionUnwrapError(RuntimeError): ...

type Option[T] = Some[T] | Null[T]
"""Type `Option[T]` represents an optional value.

See `OptionType` for more details.
"""

class OptionType[T](Pipeable):
    """OptionType is the common interface for an optional value.

    `Option[T]` is the union of `Some[T]` and `Null[T]`, and represents a value that can only have two states:

    - `Some(value)`
    - `Null()`.

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

    Example:
        ```python
        >>> from pyochain import Option, Some, Null
        >>> def divide(a: int, b: int) -> Option[int]:
        ...     if b == 0:
        ...         return Null()
        ...     return Some(a // b)
        >>>
        >>> divide(10, 2)
        Some(5)
        >>> divide(10, 0)
        NONE

        ```
    """

    def __bool__(self) -> None:
        """Prevent implicit `Some|None` value checking in boolean contexts.

        Raises:
            TypeError: Always, to prevent implicit `Some|None` value checking.

        Example:
            ```python
            >>> from pyochain import Some
            >>> x = Some(42)
            >>> bool(x)
            Traceback (most recent call last):
            ...
            TypeError: Option instances cannot be used in boolean contexts for implicit `Some|None` value checking. Use is_some() or is_none() instead.

            ```
        """

    def flatten[U](self: OptionType[Option[U]]) -> Option[U]:
        """Flattens a nested `Option`.

        Converts an `Option[Option[U]]` into an `Option[U]` by removing one level of nesting.

        Equivalent to `Option.and_then(lambda x: x)`.

        Returns:
            Option[U]: The flattened option.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some(Some(42)).flatten()
            Some(42)
            >>> Some(NONE).flatten()
            NONE
            >>> NONE.flatten()
            NONE

            ```
        """

    @overload
    def map_star[R](
        self: Option[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], R],  # pyright: ignore[reportExplicitAny]
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
        self: OptionType[U], func: Callable[..., R]
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
            >>> from pyochain import Some, NONE
            >>> Some((2, 3)).map_star(lambda x, y: x + y)
            Some(5)
            >>> NONE.map_star(lambda x, y: x + y)
            NONE

            ```
        """

    @overload
    def and_then_star[R](
        self: Option[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], Option[R]],  # pyright: ignore[reportExplicitAny]
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
        self: OptionType[U], func: Callable[..., Option[R]]
    ) -> Option[R]:
        """Calls a function if the option is `Some`, unpacking the iterable into the function.

        Args:
            func (Callable[..., Option[R]]): The function to call with the unpacked `Some` value.

        Returns:
            Option[R]: The result of the function if `Some`, otherwise `None`.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some((2, 3)).and_then_star(lambda x, y: Some(x + y))
            Some(5)
            >>> NONE.and_then_star(lambda x, y: Some(x + y))
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
            >>> from pyochain import Some, NONE
            >>> Some(42).ne(Some(21))
            True
            >>> Some(42).ne(Some(42))
            False
            >>> Some(42).ne(NONE)
            True
            >>> NONE.ne(NONE)
            False

            ```
        """

    @override
    def __eq__(self, other: object) -> bool:
        """Checks if this `Option` and *other* are equal.

        A plain Python `None` is considered equal to a `pyochain.Null` instance.

        Args:
            other (object): The other object to compare with.

        Returns:
            bool: `True` if both instances are equal, `False` otherwise.

        See Also:
            - `Option.eq` for a type-safe, performant version that only accepts `Option[T]` instances.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some(42) == Some(42)
            True
            >>> Some(42) == Some(21)
            False
            >>> Some(42) == NONE
            False
            >>> NONE == NONE
            True
            >>> NONE == None
            True
            >>> Some(42) == 42
            False

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
            >>> from pyochain import Some, NONE
            >>> Some(42).eq(Some(42))
            True
            >>> Some(42).eq(Some(21))
            False
            >>> Some(42).eq(NONE)
            False
            >>> NONE.eq(NONE)
            True

            ```
        """

    def is_some(self) -> bool:
        """Returns `True` if the option is a `Some` value.

        Returns:
            bool: `True` if the option is a `Some` variant, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> x: Option[int] = Some(2)
            >>> x.is_some()
            True
            >>> y: Option[int] = NONE
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
            >>> from pyochain import Some, NONE
            >>> x = Some(2)
            >>> x.is_some_and(lambda x: x > 1)
            True

            >>> x = Some(0)
            >>> x.is_some_and(lambda x: x > 1)
            False
            >>> x = NONE
            >>> x.is_some_and(lambda x: x > 1)
            False
            >>> x = Some("hello")
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
            >>> from pyochain import Some, NONE
            >>> x: Option[int] = Some(2)
            >>> x.is_none()
            False
            >>> y: Option[int] = NONE
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
            >>> from pyochain import Some, NONE
            >>> Some(2).is_none_or(lambda x: x > 1)
            True
            >>> Some(0).is_none_or(lambda x: x > 1)
            False
            >>> NONE.is_none_or(lambda x: x > 1)
            True
            >>> Some("hello").is_none_or(lambda x: len(x) > 1)
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
            >>> from pyochain import Some
            >>> Some("car").unwrap()
            'car'

            ```
            ```python
            >>> from pyochain import NONE
            >>> NONE.unwrap()
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
            >>> from pyochain import Some, NONE
            >>> Some("value").expect("fruits are healthy")
            'value'
            >>> NONE.expect("fruits are healthy")
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
            >>> from pyochain import Some, NONE
            >>> Some("car").unwrap_or("bike")
            'car'
            >>> NONE.unwrap_or("bike")
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
            >>> from pyochain import Some, NONE
            >>> k = 10
            >>> Some(4).unwrap_or_else(lambda: 2 * k)
            4
            >>> NONE.unwrap_or_else(lambda: 2 * k)
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
            >>> from pyochain import Some, NONE
            >>> Some("Hello, World!").map(len)
            Some(13)
            >>> NONE.map(len)
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
            >>> from pyochain import Some, NONE
            >>> Some(2).and_(NONE)
            NONE
            >>> NONE.and_(Some("foo"))
            NONE
            >>> Some(2).and_(Some("foo"))
            Some('foo')
            >>> NONE.and_(NONE)
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
            >>> from pyochain import Some, NONE
            >>> Some(2).or_(NONE)
            Some(2)
            >>> NONE.or_(Some(100))
            Some(100)
            >>> Some(2).or_(Some(100))
            Some(2)
            >>> NONE.or_(NONE)
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
            >>> from pyochain import Some, NONE
            >>> def sq(x: int) -> Option[int]:
            ...     return Some(x * x)
            >>> def nope(x: int) -> Option[int]:
            ...     return NONE
            >>> Some(2).and_then(sq).and_then(sq)
            Some(16)
            >>> Some(2).and_then(sq).and_then(nope)
            NONE
            >>> Some(2).and_then(nope).and_then(sq)
            NONE
            >>> NONE.and_then(sq).and_then(sq)
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
            >>> from pyochain import Some, NONE
            >>> def nobody() -> Option[str]:
            ...     return NONE
            >>> def vikings() -> Option[str]:
            ...     return Some("vikings")
            >>> Some("barbarians").or_else(vikings)
            Some('barbarians')
            >>> NONE.or_else(vikings)
            Some('vikings')
            >>> NONE.or_else(nobody)
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
            >>> from pyochain import Some, NONE
            >>> Some(1).ok_or("fail")
            Ok(1)
            >>> NONE.ok_or("fail")
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
            >>> from pyochain import Some, NONE
            >>> Some(1).ok_or_else(lambda: "fail")
            Ok(1)
            >>> NONE.ok_or_else(lambda: "fail")
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
            >>> from pyochain import Some, NONE
            >>> Some(2).map_or(0, lambda x: x * 10)
            20
            >>> NONE.map_or(0, lambda x: x * 10)
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
            >>> from pyochain import Some, NONE
            >>> Some(2).map_or_else(lambda: 0, lambda x: x * 10)
            20
            >>> NONE.map_or_else(lambda: 0, lambda x: x * 10)
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
            >>> from pyochain import Some, NONE
            >>>
            >>> def is_even(n: int) -> bool:
            ...     return n % 2 == 0
            >>>
            >>> NONE.filter(is_even)
            NONE
            >>> Some(3).filter(is_even)
            NONE
            >>> Some(4).filter(is_even)
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
            >>> from pyochain import Some, NONE
            >>> Some(42).iter().next()
            Some(42)
            >>> NONE.iter().next()
            NONE

            ```
        """
    @override
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
            >>> from pyochain import Some, NONE, Vec
            >>> seen = Vec[int](())
            >>> Some(2).inspect(lambda x: seen.append(x))
            Some(2)
            >>> seen
            Vec(2)
            >>> NONE.inspect(lambda x: seen.append(x))
            NONE
            >>> seen
            Vec(2)

            ```
        """

    def unzip[U](self: OptionType[tuple[T, U]]) -> tuple[Option[T], Option[U]]:
        """Unzips an `Option` of a tuple into a tuple of `Option`s.

        If the option is `Some((a, b))`, this method returns `(Some(a), Some(b))`.
        If the option is `NONE`, it returns `(NONE, NONE)`.

        Returns:
            tuple[Option[T], Option[U]]: A tuple containing two options.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some((1, "a")).unzip()
            (Some(1), Some('a'))
            >>> NONE.unzip()
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
            >>> from pyochain import Some, NONE
            >>> Some(1).zip(Some("a"))
            Some((1, 'a'))
            >>> Some(1).zip(NONE)
            NONE
            >>> NONE.zip(Some("a"))
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
            >>> from pyochain import Some, NONE
            >>>
            >>> @dataclass
            ... class Point:
            ...     x: float
            ...     y: float
            >>>
            >>> x = Some(17.5)
            >>> y = Some(42.7)
            >>> x.zip_with(y, Point)
            Some(Point(x=17.5, y=42.7))
            >>> x.zip_with(NONE, Point)
            NONE
            >>> NONE.zip_with(y, Point)
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
            >>> from pyochain import Some, NONE
            >>> s12 = Some(12)
            >>> s17 = Some(17)
            >>>
            >>> def add(a: int, b: int) -> int:
            ...     return a + b
            >>>
            >>> s12.reduce(s17, add)
            Some(29)
            >>> s12.reduce(NONE, add)
            Some(12)
            >>> NONE.reduce(s17, add)
            Some(17)
            >>> NONE.reduce(NONE, add)
            NONE

            ```
        """

    def transpose[E](self: OptionType[Result[T, E]]) -> Result[Option[T], E]:
        """Transposes an `Option` of a `Result` into a `Result` of an `Option`.

        The mapping is as follows:

        - `Some(Ok[T])` is mapped to `Ok(Some[T])`
        - `Some(Err[E])` is mapped to `Err[E]`
        - `NONE` is mapped to `Ok(NONE)`

        Returns:
            Result[Option[T], E]: The transposed result.

        Example:
            ```python
            >>> from pyochain import Some, Ok, Err, NONE
            >>> Some(Ok(5)).transpose()
            Ok(Some(5))
            >>> NONE.transpose()
            Ok(NONE)
            >>> Some(Err("error")).transpose()
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
            >>> from pyochain import Some, NONE
            >>> Some(2).xor(NONE)
            Some(2)
            >>> NONE.xor(Some(2))
            Some(2)
            >>> Some(2).xor(Some(2))
            NONE
            >>> NONE.xor(NONE)
            NONE

            ```
        """
    def unwrap_or_none(self) -> T | None:
        """Returns the contained `Some` value or `None`.

        This is a convenience method for interoperability with APIs that use `None` to represent the absence of a value,

        e.g. when interacting with standard Python libraries, or external dependencies.

        This is **NOT** the recommended use for handling `Option` in any code that can be controlled, as it defeats the purpose of using `Option` for explicit handling of optional values.

        Returns:
            T | None: The contained `Some` value or `None`.

        Example:
            ```python
            >>> from pyochain import Option, Some, NONE
            >>> NONE.unwrap_or_none() is None
            True
            >>> Some(42).unwrap_or_none()
            42

            ```
        """

@final
class Some[T](OptionType[T]):
    """Option variant representing the presence of a value.

    For more documentation, see the `Option[T]` class.

    Attributes:
        value (T): The contained value.

    Example:
        ```python
        >>> from pyochain import Some
        >>> Some(42)
        Some(42)

        ```
    """

    value: T
    __match_args__ = ("value",)
    # Hack to immediately handle it as an "enum".
    @overload
    def __new__[E](cls, value: Result[T, E]) -> Option[Result[T, E]]: ...
    @overload
    def __new__(cls, value: Null[T]) -> Option[T]: ...
    @overload
    def __new__(cls, value: T) -> Option[T]: ...

@final
class Null[T](OptionType[T]):
    """Option variant representing the absence of a value.

    This class or `NONE` can be used interchangeably, as calling `Null()` will always return the singleton instance `NONE`.

    For pattern matching, you must use `Null`, as `NONE` isn't special cased by type checkers the same way python `None` is, and thus can't be narrowed to `Null`.

    For more documentation, see the `Option[T]` class.

    Example:
        ```python
        >>> from pyochain import Null, NONE, Some
        >>> Null() is NONE
        True
        >>> def is_none(x: Option[int]) -> bool:
        ...     match x:
        ...         case Null():
        ...             return True
        ...         case Some(_):
        ...             return False
        >>> is_none(NONE)
        True
        >>> is_none(Some(42))
        False
        >>> is_none(Null())
        True

        ```
    """

NONE: Final[Null[Any]] = ...  # pyright: ignore[reportAny, reportExplicitAny]
"""Singleton instance representing the absence of a value.

This is the only instance of `Null` who exists, and is similar to the logic used by `None` in standard Python.

This allows you to improve performance by avoiding unnecessary calls to `Null::__new__`.

Warning:
    Reassigning this variable is not recommended.
"""

def option[T](value: T | None) -> Option[T]:
    """Creates an `Option[V]` from a value that may be `None`.

    When calling `Option(value)`, this method automatically redirects to:
    - `Some(value)` if the value is not `None`
    - `NONE` if the value is `None`

    Args:
        value (T | None): The value to convert into an `Option[T]`.

    Returns:
        Option[T]: `Some(value)` if the value is not `None`, otherwise `NONE`.

    Example:
        ```python
        >>> from pyochain import option
        >>> option(42)
        Some(42)
        >>> option(None)
        NONE

        ```
    """

def then_if_true[T](value: T, *, predicate: Callable[[T], bool]) -> Option[T]:
    """Creates an `Option[T]` based on a **predicate** condition on the provided **value**.

    Args:
        value (T): The value to wrap in `Some` if the condition is `True`.
        predicate (Callable[[T], bool]): The condition to evaluate.

    Returns:
        Option[T]: `Some(value)` if the condition is `True`, otherwise `NONE`.

    Example:
        ```python
        >>> from pyochain import then_if_true
        >>> then_if_true(42, predicate=lambda x: x == 42)
        Some(42)
        >>> then_if_true(21, predicate=lambda x: x == 42)
        NONE
        >>> from pathlib import Path
        >>> readme_path = then_if_true(Path("README.md"), predicate=Path.exists).map(
        ...     str
        ... )
        >>> readme_path
        Some('README.md')

        ```
    """

def then_if_some[T](value: T) -> Option[T]:
    """Creates an `Option[T]` based on the truthiness of a value.

    Args:
        value (T): The value to evaluate.

    Returns:
        Option[T]: `Some(value)` if the value is truthy, otherwise `NONE`.

    Example:
        ```python
        >>> from pyochain import then_if_some
        >>> then_if_some(42)
        Some(42)
        >>> then_if_some(0)
        NONE
        >>> then_if_some("hello")
        Some('hello')
        >>> then_if_some("")
        NONE
        >>> then_if_some(())  # Empty sequence is falsy
        NONE

        ```
    """

class ResultUnwrapError(RuntimeError): ...

type Result[T, E] = Ok[T, E] | Err[T, E]
"""Type union representing the two variants of `Result`, `Ok` and `Err`.

See the `ResultType` Protocol for documentation on the methods available on `Result`, and the behavior of each variant.
"""

@type_check_only
class ResultType[T, E](Pipeable, Protocol):
    """This is the base Protocol defined for returning and propagating errors.

    `Result[T, E]` is a the type union of the two possibles variants of the Protocol:

    - `Ok[T, E]`, representing success and containing a value
    - `Err[T, E]`, representing error and containing an error value

    Functions return `Result` whenever errors are expected and recoverable.

    For example, I/O or web requests can fail for many reasons, and using `Result` forces the caller to handle the possibility of failure.

    This is directly inspired by Rust's `Result` type, and provides similar functionality for error handling in Python.

    Note:
        Due to Python typing nature, we need to separate both the Protocol definition (`ResultType`), and the type union (`Result`), which is the public facing type that users will interact with.

        This separation allows type checkers to flag exhaustive handling of both variants, in `match` statements notably, while avoiding duplicated docstrings and method definitions.

    Warning:
        Do not try to instanciate this class, as it don't exist at runtime.

        `Result` does in fact exist in the namespace, but it's an empty `Rust` struct,

        and your type checker will warn you in any case because a `type Result = ...` is not supposed to be instanciable.

    Example:
        ```python
        >>> from pyochain import Err, Ok, Result
        >>>
        >>> def is_positive(x: int) -> Result[str, ValueError]:
        ...     if x > 0:
        ...         return Ok(f"Value is {x}")
        ...     msg = f"{x} is not positive"
        ...     return Err(ValueError(msg))
        >>>
        >>> def handle_variant(x: Result[str, ValueError]) -> str:
        ...     match x:
        ...         case Ok(value):
        ...             return f"Success: {value}"
        ...         case Err(error):
        ...             return f"Failure: {error}"
        >>>
        >>> is_positive(5).map(lambda s: s.upper()).into(handle_variant)
        'Success: VALUE IS 5'
        >>> is_positive(-3).map(lambda s: s.upper()).into(handle_variant)
        'Failure: -3 is not positive'

        ```
    """

    def swap(self) -> Result[E, T]:
        """Swaps the `Ok` and `Err` variants.

        Converts an `Ok[T]` into an `Err[T]` and an `Err[E]` into an `Ok[E]`.

        Returns:
            Result[E, T]: The swapped result.

        Example:
            ```python
            >>> from pyochain import Ok, Err
            >>> Ok(2).swap()
            Err(2)
            >>> Err("error").swap()
            Ok('error')

            ```
        """
    def flatten[T1, E1, E2](self: ResultType[Result[T1, E1], E2]) -> Result[T1, E1]:
        """Flattens a nested `Result`.

        Converts from `Result[Result[T, E], E]` to `Result[T, E]`.

        Equivalent to calling `Result.and_then(lambda x: x)`, but more convenient when there's no need to process the inner `Ok` value.

        Returns:
            Result[T, E]: The flattened result.

        Example:
            ```python
            >>> from pyochain import Result, Ok, Err
            >>> nested_ok: Result[Result[int, str], str] = Ok(Ok(2))
            >>> nested_ok.flatten()
            Ok(2)
            >>> nested_err: Result[Result[int, str], str] = Ok(Err("inner error"))
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
            >>> from pyochain import Ok, Err
            >>> Ok(7).iter().next()
            Some(7)
            >>> Err("nothing!").iter().next()
            NONE

            ```
        """

    @overload
    def map_star[R](
        self: Result[tuple[Any], E],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], R],  # pyright: ignore[reportExplicitAny]
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
        self: ResultType[U, E],
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
            >>> from pyochain import Ok, Err
            >>> Ok((2, 3)).map_star(lambda x, y: x + y)
            Ok(5)
            >>> Err("error").map_star(lambda x, y: x + y)
            Err('error')

            ```
        """

    @overload
    def and_then_star[R](
        self: Result[tuple[Any], E],  # pyright: ignore[reportExplicitAny]
        func: Callable[[Any], Result[R, E]],  # pyright: ignore[reportExplicitAny]
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
        self: ResultType[U, E],
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
            >>> from pyochain import Ok, Err, Result
            >>> def to_str(x: int, y: int) -> Result[str, str]:
            ...     return Ok(f"{x},{y}")
            >>> Ok((2, 3)).and_then_star(to_str)
            Ok('2,3')
            >>> Err("error").and_then_star(to_str)
            Err('error')

            ```
        """

    def is_ok(self) -> bool:
        """Returns `True` if the result is `Ok`.

        Returns:
            bool: `True` if the result is an `Ok` variant, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Ok, Err, Result
            >>> x: Result[int, str] = Ok(2)
            >>> x.is_ok()
            True
            >>> y: Result[int, str] = Err("Some error message")
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
            >>> from pyochain import Ok, Err, Result
            >>> x: Result[int, str] = Ok(2)
            >>> x.is_err()
            False
            >>> y: Result[int, str] = Err("Some error message")
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
            >>> from pyochain import Ok
            >>> Ok(2).unwrap()
            2

            ```
            ```python
            >>> from pyochain import Err
            >>> Err("emergency failure").unwrap()
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
            >>> from pyochain import Err
            >>> Err("emergency failure").unwrap_err()
            'emergency failure'

            ```
            ```python
            >>> from pyochain import Ok
            >>> Ok(2).unwrap_err()
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
            >>> from pyochain import Ok, Err
            >>> k = 21
            >>> Ok("foo").map_or_else(len, lambda e: k * 2)
            3
            >>> Err("bar").map_or_else(len, lambda e: k * 2)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).expect("No error")
            2
            >>> Err("emergency failure").expect("Testing expect")
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
            >>> from pyochain import Err, Ok
            >>> Err("emergency failure").expect_err("Testing expect_err")
            'emergency failure'
            >>> Ok(10).expect_err("Testing expect_err")
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).unwrap_or(10)
            2
            >>> Err("error").unwrap_or(10)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).unwrap_or_else(len)
            2
            >>> Err("foo").unwrap_or_else(len)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).map(lambda x: x * 2)
            Ok(4)
            >>> Err("error").map(lambda x: x * 2)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).map_err(len)
            Ok(2)
            >>> Err("foo").map_err(len)
            Err(3)

            ```
        """
    @override
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
            >>> from pyochain import Ok, Vec
            >>> seen = Vec[int](())
            >>> Ok(2).inspect(lambda x: seen.append(x))
            Ok(2)
            >>> seen
            Vec(2)

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
            >>> from pyochain import Err, Vec
            >>> seen = Vec[str](())
            >>> Err("oops").inspect_err(lambda e: seen.append(e))
            Err('oops')
            >>> seen
            Vec('oops')

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
            >>> from pyochain import Ok, Err
            >>> x = Ok(2)
            >>> y = Err("late error")
            >>> x.and_(y)
            Err('late error')
            >>> x = Err("early error")
            >>> y = Ok("foo")
            >>> x.and_(y)
            Err('early error')

            >>> x = Err("not a 2")
            >>> y = Err("late error")
            >>> x.and_(y)
            Err('not a 2')

            >>> x = Ok(2)
            >>> y = Ok("different result type")
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
            >>> from pyochain import Ok, Err, Result
            >>> def to_str(x: int) -> Result[str, str]:
            ...     return Ok(str(x))
            >>> Ok(2).and_then(to_str)
            Ok('2')
            >>> Err("error").and_then(to_str)
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
            >>> from pyochain import Ok, Err, Result
            >>> def fallback(e: str) -> Result[int, str]:
            ...     return Ok(len(e))
            >>> Ok(2).or_else(fallback)
            Ok(2)
            >>> Err("foo").or_else(fallback)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).ok()
            Some(2)
            >>> Err("error").ok()
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).err()
            NONE
            >>> Err("error").err()
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).is_ok_and(lambda x: x > 1)
            True
            >>> Ok(0).is_ok_and(lambda x: x > 1)
            False
            >>> Err("err").is_ok_and(lambda x: x > 1)
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
            >>> from pyochain import Err, Ok
            >>> Err("foo").is_err_and(lambda e: len(e) == 3)
            True
            >>> Err("bar").is_err_and(lambda e: e == "baz")
            False
            >>> Ok(2).is_err_and(lambda e: True)
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).map_or(10, lambda x: x * 2)
            4
            >>> Err("err").map_or(10, lambda x: x * 2)
            10

            ```
        """

    def transpose(self: ResultType[Option[T], E]) -> Option[Result[T, E]]:
        """Transposes a Result containing an Option into an Option containing a Result.

        Can only be called if the inner type is `Option[T, E]`.

        The mapping is as follows:

        - `Ok(Some(v))` becomes `Some(Ok(v))`
        - `Ok(NONE)` becomes `NONE`
        - `Err(e)` becomes `Some(Err(e))`

        Returns:
            Option[Result[T, E]]: Option containing a Result or NONE.

        Example:
            ```python
            >>> from pyochain import Ok, Err, Some, NONE
            >>> Ok(Some(2)).transpose()
            Some(Ok(2))
            >>> Ok(NONE).transpose()
            NONE
            >>> Err("err").transpose()
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
            >>> from pyochain import Ok, Err
            >>> Ok(2).or_(Err("late error"))
            Ok(2)
            >>> Err("early error").or_(Ok(2))
            Ok(2)
            >>> Err("not a 2").or_(Err("late error"))
            Err('late error')
            >>> Ok(2).or_(Ok(100))
            Ok(2)

            ```
        """

@final
class Ok[T, E](ResultType[T, E]):
    """Represents a successful value.

    One of the two variants of `Result[T, E]`, where `T` is the type of the value in `Ok`.

    For more documentation, see the `ResultType[T, E]` Protocol.

    Attributes:
        value (T): The contained successful value.
    """

    __match_args__ = ("value",)

    value: T
    # NOTE: this is an hack to avoid errors by immediatly casting `E` as `Any`, thus avoiding any type errors with incompatible types.
    @overload
    def __new__(cls, value: Result[T, E]) -> Result[Result[T, E], Any]: ...  # pyright: ignore[reportExplicitAny]
    @overload
    def __new__(cls, value: Option[T]) -> Result[Option[T], Any]: ...  # pyright: ignore[reportExplicitAny]
    @overload
    def __new__(cls, value: T) -> Result[T, Any]: ...  # pyright: ignore[reportExplicitAny]

@final
class Err[T, E](ResultType[T, E]):
    """Represents an error value.

    One of the two variants of `Result[T, E]`, where `E` is the type of the value in `Err`.

    For more documentation, see the `ResultType[T, E]` Protocol.

    Attributes:
        error (E): The contained error value.
    """

    __match_args__ = ("error",)

    error: E
    # NOTE: same hack as in `Ok` for type errors
    @overload
    def __new__(cls, error: Result[T, E]) -> Result[Any, Result[T, E]]: ...  # pyright: ignore[reportExplicitAny]
    @overload
    def __new__(cls, error: Option[E]) -> Result[Any, Option[E]]: ...  # pyright: ignore[reportExplicitAny]
    @overload
    def __new__(cls, error: E) -> Result[Any, E]: ...  # pyright: ignore[reportExplicitAny]
