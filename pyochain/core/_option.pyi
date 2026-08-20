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
        from pyochain import Option, Some, Null

        def divide(a: int, b: int) -> Option[int]:
            if b == 0:
                return Null()
            return Some(a // b)

        assert divide(10, 2) == Some(5)
        assert divide(10, 0).is_none()
        ```
    """

    def __bool__(self) -> None:
        """Prevent implicit `Some|None` value checking in boolean contexts.

        Always raises `TypeError` to prevent implicit `Some|None` value checking, as the `Option` truthiness is ambiguous.

        Are we checking the presence or absence of the value, or the truthiness of the contained value?

        Use `Option::{is_some, is_none, filter, map_if, or_else}`, and others alike for combining control flow with `Option` values.

        Example:
            ```python
            from pyochain import Some
            import pytest

            x = Some(42)
            with pytest.raises(TypeError):
                bool(x)
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
            from pyochain import Some, NONE

            assert Some(42) == Some(42)
            assert Some(42) != Some(21)
            assert Some(42) != NONE
            assert NONE == NONE
            assert Some(42) != 42
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
            from pyochain import Some, NONE

            assert Some(Some(42)).flatten() == Some(42)
            assert Some(NONE).flatten().is_none()
            assert NONE.flatten().is_none()
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
            from pyochain import Some, NONE

            assert Some((2, 3)).map_star(lambda x, y: x + y) == Some(5)
            assert NONE.map_star(lambda x, y: x + y).is_none()
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
            from pyochain import Some, NONE

            assert Some((2, 3)).and_then_star(lambda x, y: Some(x + y)) == Some(5)
            assert NONE.and_then_star(lambda x, y: Some(x + y)).is_none()
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
            from pyochain import Some, NONE

            assert Some(42).ne(Some(21))
            assert not Some(42).ne(Some(42))
            assert Some(42).ne(NONE)
            assert not NONE.ne(NONE)
            ```
        """

    def eq(self, other: Option[object]) -> bool:
        """Checks if two `Option[T]` instances are equal.

        Note:
            This method behave similarly to `__eq__`, but only accepts `Option[T]` instances as argument.

            This avoids runtime isinstance checks (we check for boolean `is_some()`, which is a simple function call), and is more type-safe.

        Args:
            other (Option[object]): The other `Option[T]` instance to compare with.

        Returns:
            bool: `True` if both instances are equal, `False` otherwise.

        Example:
            ```python
            from pyochain import Some, NONE

            assert Some(42).eq(Some(42))
            assert not Some(42).eq(Some(21))
            assert not Some(42).eq(NONE)
            assert NONE.eq(NONE)
            ```
        """

    def is_some(self) -> bool:
        """Returns `True` if the option is a `Some` value.

        Returns:
            bool: `True` if the option is a `Some` variant, `False` otherwise.

        Example:
            ```python
            from pyochain import Some, NONE

            assert Some(2).is_some()
            assert not NONE.is_some()
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
            from pyochain import Some, NONE

            x = Some(2)
            assert x.is_some_and(lambda x: x > 1)

            x = Some(0)
            assert not x.is_some_and(lambda x: x > 1)

            x = NONE
            assert not x.is_some_and(lambda x: x > 1)

            x = Some("hello")
            assert x.is_some_and(lambda x: len(x) > 1)
            ```
        """

    def is_none(self) -> bool:
        """Returns `True` if the option is a `None` value.

        Returns:
            bool: `True` if the option is a `_None` variant, `False` otherwise.

        Example:
            ```python
            from pyochain import Some, NONE

            x = Some(2)
            assert not x.is_none()
            y = NONE

            assert y.is_none()
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
            from pyochain import Some, NONE

            assert Some(2).is_none_or(lambda x: x > 1)
            assert not Some(0).is_none_or(lambda x: x > 1)
            assert NONE.is_none_or(lambda x: x > 1)
            assert Some("hello").is_none_or(lambda x: len(x) > 1)
            ```
        """

    def unwrap(self) -> T:
        """Returns the contained `Some` value.

        raises `OptionUnwrapError` if the option is `None`.

        Returns:
            T: The contained `Some` value.

        Example:
            ```python
            from pyochain import Some, NONE, OptionUnwrapError

            assert Some("car").unwrap() == "car"

            try:
                NONE.unwrap()
            except OptionUnwrapError as e:
                assert str(e) == "called `unwrap` on a `None`"
            ```
        """

    def expect(self, msg: str) -> T:
        """Returns the contained `Some` value.

        Raises an exception with a provided message if the value is `None`.

        Args:
            msg (str): The message to include in the exception if the result is `None`.

        Returns:
            T: The contained `Some` value.

        Example:
            ```python
            from pyochain import Some, NONE, OptionUnwrapError

            assert Some("value").expect("fruits are healthy") == "value"

            try:
                NONE.expect("fruits are healthy")
            except OptionUnwrapError as e:
                assert str(e) == "fruits are healthy (called `expect` on a `None`)"
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
            from pyochain import Some, NONE

            assert Some("car").unwrap_or("bike") == "car"
            assert NONE.unwrap_or("bike") == "bike"
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
            from pyochain import Some, NONE

            k = 10

            assert Some(4).unwrap_or_else(lambda: 2 * k) == 4
            assert NONE.unwrap_or_else(lambda: 2 * k) == 20
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
            from pyochain import Some, NONE

            assert Some("Hello, World!").map(len) == Some(13)
            assert NONE.map(len).is_none()
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
            from pyochain import Some, NONE

            assert Some(2).and_(NONE).is_none()
            assert NONE.and_(Some("foo")).is_none()
            assert Some(2).and_(Some("foo")) == Some("foo")
            assert NONE.and_(NONE).is_none()
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
            from pyochain import Some, NONE

            assert Some(2).or_(NONE) == Some(2)
            assert NONE.or_(Some(100)) == Some(100)
            assert Some(2).or_(Some(100)) == Some(2)
            assert NONE.or_(NONE).is_none()
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
            from pyochain import Some, NONE, Option

            def sq(x: int) -> Option[int]:
                return Some(x * x)

            def nope(x: int) -> Option[int]:
                return NONE

            assert Some(2).and_then(sq).and_then(sq) == Some(16)
            assert Some(2).and_then(sq).and_then(nope).is_none()
            assert Some(2).and_then(nope).and_then(sq).is_none()
            assert NONE.and_then(sq).and_then(sq).is_none()
            ```
        """

    def or_else[S](self, f: Callable[[], Option[S]]) -> Option[T | S]:
        """Returns the `Option[T]` if it contains a value, otherwise calls a function and returns the result.

        Args:
            f (Callable[[], Option[S]]): The function to call if the option is `None`.

        Returns:
            Option[T | S]: The original `Option` if it is `Some`, otherwise the result of the function.

        Example:
            ```python
            from pyochain import Some, NONE, Option

            def nobody() -> Option[str]:
                return NONE

            def vikings() -> Option[str]:
                return Some("vikings")

            assert Some("barbarians").or_else(vikings) == Some("barbarians")
            assert NONE.or_else(vikings) == Some("vikings")
            assert NONE.or_else(nobody).is_none()
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
            from pyochain import Some, NONE, Ok

            assert Some(1).ok_or("fail").unwrap() == 1
            assert NONE.ok_or("fail").unwrap_err() == "fail"
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
            from pyochain import Some, NONE, Ok, Err

            assert Some(1).ok_or_else(lambda: "fail").unwrap() == 1
            assert NONE.ok_or_else(lambda: "fail").unwrap_err() == "fail"
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
            from pyochain import Some, NONE

            assert Some(2).map_or(0, lambda x: x * 10) == 20
            assert NONE.map_or(0, lambda x: x * 10) == 0
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
            from pyochain import Some, NONE

            assert Some(2).map_or_else(lambda: 0, lambda x: x * 10) == 20
            assert NONE.map_or_else(lambda: 0, lambda x: x * 10) == 0
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
            from pyochain import Some, NONE

            def is_even(n: int) -> bool:
                return n % 2 == 0

            assert NONE.filter(is_even).is_none()
            assert Some(3).filter(is_even).is_none()
            assert Some(4).filter(is_even) == Some(4)
            ```
        """

    def iter(self) -> PyoIterator[T]:
        """Creates an `Iterator` over the optional value.

        - If the option is `Some(value)`, the iterator yields `value`.
        - If the option is `NONE`, the iterator yields nothing.

        Equivalent to `Iter(self.unwrap())` if `Some`, or `Iter()` if `NONE`.

        Returns:
            PyoIterator[T]: An `Iterator` over the optional value.

        Example:
            ```python
            from pyochain import Some, NONE, Iter

            assert Some(42).iter().next() == Some(42)
            assert NONE.iter().next().is_none()
            assert Iter(42).next() == Some(42).iter().next()
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
            from pyochain import Some, NONE, Vec

            seen = Vec[int]([])

            assert Some(2).inspect(lambda x: seen.append(x)) == Some(2)
            assert seen == Vec(2)

            assert NONE.inspect(lambda x: seen.append(x)).is_none()
            assert seen == Vec(2)
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
            from pyochain import Some, NONE

            assert Some((1, "a")).unzip() == (Some(1), Some("a"))
            assert NONE.unzip() == (NONE, NONE)
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
            from pyochain import Some, NONE

            assert Some(1).zip(Some("a")) == Some((1, "a"))
            assert Some(1).zip(NONE).is_none()
            assert NONE.zip(Some("a")).is_none()
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
            from dataclasses import dataclass
            from pyochain import Some, NONE

            @dataclass
            class Point:
                x: float
                y: float

            x = Some(17.5)
            y = Some(42.7)

            assert x.zip_with(y, Point) == Some(Point(x=17.5, y=42.7))
            assert x.zip_with(NONE, Point).is_none()
            assert NONE.zip_with(y, Point).is_none()
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
            from pyochain import Some, NONE

            s12 = Some(12)
            s17 = Some(17)

            def add(a: int, b: int) -> int:
                return a + b

            assert s12.reduce(s17, add) == Some(29)
            assert s12.reduce(NONE, add) == Some(12)
            assert NONE.reduce(s17, add) == Some(17)
            assert NONE.reduce(NONE, add).is_none()

            def concat(a: str, b: str) -> str:
                return a + b

            a = Some("Hello, ").reduce(Some("World!"), concat)
            assert a == Some("Hello, World!")
            b = Some("I am ").reduce(Some(26), lambda a, b: a + str(b))
            assert b == Some("I am 26")
            ```
        """

    def transpose[S, E](self: OptionType[Result[S, E]]) -> Result[Option[S], E]:
        """Transposes an `Option` of a `Result` into a `Result` of an `Option`.

        The mapping is as follows:

        - `Some(Ok[T])` is mapped to `Ok(Some[T])`
        - `Some(Err[E])` is mapped to `Err[E]`
        - `NONE` is mapped to `Ok(NONE)`

        Returns:
            Result[Option[S], E]: The transposed result.

        Example:
            ```python
            from pyochain import Some, Ok, Err, NONE

            assert Some(Ok(5)).transpose().unwrap().unwrap() == 5
            assert NONE.transpose().unwrap().is_none()
            assert Some(Err("error")).transpose().unwrap_err() == "error"
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
            from pyochain import Some, NONE

            assert Some(2).xor(NONE).unwrap() == 2
            assert NONE.xor(Some(2)).unwrap() == 2
            assert Some(2).xor(Some(2)).is_none()
            assert NONE.xor(NONE).is_none()
            assert Some("hello").xor(Some(1)).is_none()
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
            from pyochain import Option, Some, NONE

            assert NONE.unwrap_or_none() is None
            assert Some(42).unwrap_or_none() == 42
            ```
        """

@final
class Some[T](OptionType[T]):
    """Option variant representing the presence of a value.

    For more documentation, see the `Option[T]` class.

    Example:
        ```python
        from pyochain import Some

        assert Some(42).pipe(repr) == "Some(42)"
        assert Some("hello").pipe(repr) == "Some('hello')"

        match Some(42):
            case Some(value):
                assert value == 42
            case Null():
                assert False, "This should never happen"
        ```
    """

    value: Final[T]
    """Final[T]: The contained value."""
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
        from pyochain import Null, NONE, Some, Option

        assert Null() is NONE

        def is_none(x: Option[int]) -> bool:
            match x:
                case Null():
                    return True
                case Some(_):
                    return False

        assert is_none(NONE)
        assert not is_none(Some(42))
        assert is_none(Null())
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
        from pyochain import option, Some, NONE

        assert option(42) == Some(42)
        assert option(None).is_none()
        ```
    """
