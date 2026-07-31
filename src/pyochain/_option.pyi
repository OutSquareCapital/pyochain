from collections.abc import Callable, Iterable
from typing import Any, Concatenate, Final, final, overload, override

from pyochain import Result
from pyochain.abc import Pipe, PyoIterator

class OptionUnwrapError(RuntimeError): ...

type Option[T] = Some[T] | Null[T]
"""Type `Option[T]` represents an optional value.

See `OptionType` for more details.
"""

class OptionType[T](Pipe):
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

    def ne(self, other: Option[object]) -> bool:
        """Checks if two `Option[T]` instances are not equal.

        Args:
            other (Option[object]): The other `Option[object]` instance to compare with.

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
            [`Option::eq`][eq] for a type-safe, performant version that only accepts `Option[T]` instances.

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

    def eq(self, other: Option[object]) -> bool:
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
            >>>
            >>> x = Some(2)
            >>> x.is_some()
            True
            >>> y = NONE
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
            >>>
            >>> x = Some(2)
            >>> x.is_none()
            False
            >>> y = NONE
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

    def unwrap_or[S](self, default: S) -> T | S:
        """Returns the contained `Some` value or a provided default.

        Args:
            default (S): The value to return if the result is `None`.

        Returns:
            T | S: The contained `Some` value or the provided default.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some("car").unwrap_or("bike")
            'car'
            >>> NONE.unwrap_or("bike")
            'bike'

            ```
        """

    def unwrap_or_else[S](self, f: Callable[[], S]) -> T | S:
        """Returns the contained `Some` value or computes it from a function.

        Args:
            f (Callable[[], S]): A function that returns a default value if the result is `None`.

        Returns:
            T | S: The contained `Some` value or the result of the function.

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
    def or_[S](self, optb: Option[S]) -> Option[T | S]:
        """Returns the option if it contains a value, otherwise returns optb.

        Args:
            optb (Option[S]): The option to return if the original option is `NONE`.

        Returns:
            Option[T | S]: The original option if it is `Some`, otherwise `optb`.

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
            >>> from pyochain import Some, NONE, Option
            >>>
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

    def or_else[S](self, f: Callable[[], Option[S]]) -> Option[T | S]:
        """Returns the `Option[T]` if it contains a value, otherwise calls a function and returns the result.

        Args:
            f (Callable[[], Option[T]]): The function to call if the option is `None`.

        Returns:
            Option[T]: The original `Option` if it is `Some`, otherwise the result of the function.

        Example:
            ```python
            >>> from pyochain import Some, NONE, Option
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
        """Returns `NONE` if the option is `NONE`, otherwise calls predicate with the wrapped value.

        This function works similar to `PyoIterator::filter` in the sense that we only keep the value if it matches a predicate.

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

    def iter(self) -> PyoIterator[T]:
        """Creates an `Iterator` over the optional value.

        - If the option is `Some(value)`, the iterator yields `value`.
        - If the option is `NONE`, the iterator yields nothing.

        Equivalent to `Iter((self,))`.

        Returns:
            PyoIterator[T]: An `Iterator` over the optional value.

        Example:
            ```python
            >>> from pyochain import Some, NONE
            >>> Some(42).iter().next()
            Some(42)
            >>> NONE.iter().next()
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

    def unzip[S, U](self: OptionType[tuple[S, U]]) -> tuple[Option[S], Option[U]]:
        """Unzips an `Option` of a tuple into a tuple of `Option`s.

        If the option is `Some((a, b))`, this method returns `(Some(a), Some(b))`.
        If the option is `NONE`, it returns `(NONE, NONE)`.

        Returns:
            tuple[Option[S], Option[U]]: A tuple containing two options.

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

    def reduce[O, R](self, other: Option[O], func: Callable[[T, O], R]) -> Option[R]:
        """Reduces two options into one, using the provided function if both are Some.

        If **self** is `Some(s)` and **other** is `Some(o)`, this method returns `Some(func(s, o))`.

        Otherwise, if only one of **self** and **other** is `Some`, that value is returned.

        If both **self** and **other** are `NONE`, `NONE` is returned.

        Args:
            other (Option[O]): The second option.
            func (Callable[[T, O], R]): The function to apply to the unwrapped values.

        Returns:
            Option[R]: The resulting option after reduction.

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
            >>> def concat(a: str, b: str) -> str:
            ...     return a + b
            >>> Some("Hello, ").reduce(Some("World!"), concat)
            Some('Hello, World!')
            >>> Some("I am ").reduce(Some(26), lambda a, b: a + str(b))
            Some('I am 26')

            ```
        """

    def transpose[S, E](self: OptionType[Result[S, E]]) -> Result[Option[S], E]:
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

    def xor[O](self, optb: Option[object]) -> Option[T]:
        """Returns `Some` if exactly one of **self**, optb is `Some`, otherwise returns `NONE`.

        Args:
            optb (Option[object]): The other option to compare with.

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
            >>> Some("hello").xor(Some(1))
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

    value: Final[T]
    __match_args__ = ("value",)
    # Hack to immediately handle it as an "enum".
    @overload
    def __new__[E](cls, value: Result[T, E]) -> Option[Result[T, E]]: ...
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
        >>> from pyochain import Null, NONE, Some, Option
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

NONE: Final[Null[Any]] = ...  # pyright: ignore[reportAny]
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
