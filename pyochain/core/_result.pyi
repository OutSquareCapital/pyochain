from collections.abc import Callable
from typing import Any, Concatenate, Final, Protocol, final, overload, type_check_only

from pyochain import Option
from pyochain.abc import Pipe, PyoIterator

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
class ResultType[T, E](Pipe, Protocol):
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
        >>> is_positive(5).map(lambda s: s.upper()).pipe(handle_variant)
        'Success: VALUE IS 5'
        >>> is_positive(-3).map(lambda s: s.upper()).pipe(handle_variant)
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

        Converts from `Result[Result[T1, E1], E2]` to `Result[T1, E1]`.

        Equivalent to calling `Result.and_then(lambda x: x)`, but more convenient when there's no need to process the inner `Ok` value.

        Returns:
            Result[T1, E1]: The flattened result.

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

    def iter(self) -> PyoIterator[T]:
        """Returns an `Iterator` over the possibly contained value.

        Returns:
            PyoIterator[T]: An `Iterator` over the `Ok` value, or empty if `Err`.

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
    def map_star[U: tuple[Any, ...], R](
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
    def and_then_star[S, T1, R](
        self: Result[tuple[T1], S],
        func: Callable[[T1], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, R](
        self: Result[tuple[T1, T2], S],
        func: Callable[[T1, T2], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, R](
        self: Result[tuple[T1, T2, T3], S],
        func: Callable[[T1, T2, T3], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, R](
        self: Result[tuple[T1, T2, T3, T4], S],
        func: Callable[[T1, T2, T3, T4], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, R](
        self: Result[tuple[T1, T2, T3, T4, T5], S],
        func: Callable[[T1, T2, T3, T4, T5], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, T6, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6], S],
        func: Callable[[T1, T2, T3, T4, T5, T6], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, T6, T7, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7], S],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8], S],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9], S],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], Result[R, S]],
    ) -> Result[R, S]: ...
    @overload
    def and_then_star[S, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: Result[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], S],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], Result[R, S]],
    ) -> Result[R, S]: ...
    def and_then_star[U: tuple[Any, ...], R](
        self: Result[U, E], func: Callable[..., Result[R, E]]
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

        raises `ResultUnwrapError` if the result is `Err`.

        Returns:
            T: The contained `Ok` value.

        Example:
            ```python
            from pyochain import Ok, Err, ResultUnwrapError

            assert Ok(2).unwrap() == 2

            try:
                _ = Err(1).unwrap()
            except ResultUnwrapError as e:
                assert str(e) == "called `unwrap` on an `Err`: 1"
            ```
        """

    def unwrap_err(self) -> E:
        """Returns the contained `Err` value.

        raises `ResultUnwrapError` if the result is `Ok`.

        Returns:
            E: The contained `Err` value.

        Example:
            ```python
            >>> from pyochain import Err
            >>> Err("emergency failure").unwrap_err()
            'emergency failure'

            ```
            ```python
            from pyochain import Ok, ResultUnwrapError

            try:
                _ = Ok(2).unwrap_err()
            except ResultUnwrapError as e:
                assert str(e) == "called `unwrap_err` on Ok"
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
            from pyochain import Ok, Err

            k = 21
            assert Ok("foo").map_or_else(len, lambda e: k * 2) == 3
            assert Err("bar").map_or_else(len, lambda e: k * 2) == 42
            ```
        """

    def expect(self, msg: str) -> T:
        """Returns the contained `Ok` value.

        raises `ResultUnwrapError` with a provided message if the value is an `Err`.

        Args:
            msg (str): The message to include in the exception if the result is `Err`.

        Returns:
            T: The contained `Ok` value.

        Example:
            ```python
            from pyochain import Err, Ok, ResultUnwrapError

            assert Ok(2).expect("No error") == 2
            try:
                _ = Err(1).expect("Unexpected error")
            except ResultUnwrapError as e:
                assert str(e) == "Unexpected error: 1"
            ```
        """

    def expect_err(self, msg: str) -> E:
        """Returns the contained `Err` value.

        raises `ResultUnwrapError` with a provided message if the value is an `Ok`.

        Args:
            msg (str): The message to include in the exception if the result is `Ok`.

        Returns:
            E: The contained `Err` value.

        Example:
            ```python
            from pyochain import Err, Ok, ResultUnwrapError

            e = Err("emergency failure").expect_err("Testing expect_err")
            assert str(e) == "emergency failure"
            try:
                _ = Ok(10).expect_err("Testing expect_err")
            except ResultUnwrapError as e:
                assert str(e) == "Testing expect_err: expected Err, got Ok(10)"
            ```
        """

    def unwrap_or[D](self, default: D) -> T | D:
        """Returns the contained `Ok` value or a provided default.

        Args:
            default (D): The value to return if the result is `Err`.

        Returns:
            T | D: The contained `Ok` value or the provided default.

        Example:
            ```python
            from pyochain import Ok, Err

            assert Ok(2).unwrap_or(10) == 2
            assert Err("error").unwrap_or(10) == 10
            ```
        """

    def unwrap_or_else[**P, O](
        self, fn: Callable[Concatenate[E, P], O], *args: P.args, **kwargs: P.kwargs
    ) -> T | O:
        """Returns the contained `Ok` value or computes it from a function.

        Args:
            fn (Callable[Concatenate[E, P], O]): A function that takes the `Err` value and returns a default value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            T | O: The contained `Ok` value or the result of the function.

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

    def and_[O, U](self, res: Result[U, O]) -> Result[U, E | O]:
        """Returns `res` if the result is `Ok`, otherwise returns the `Err` value.

        This is often used for chaining operations that might fail.

        Args:
            res (Result[U, O]): The result to return if the original result is `Ok`.

        Returns:
            Result[U, E | O]: `res` if the original result is `Ok`, otherwise the original `Err`.

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

    def and_then[**P, R, NE](
        self,
        fn: Callable[Concatenate[T, P], Result[R, object]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[R, E]:
        """Calls a function if the result is `Ok`, otherwise returns the `Err` value.

        This is often used for chaining operations that might fail.

        Args:
            fn (Callable[Concatenate[T, P], Result[R, object]]): The function to call with the `Ok` value.
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

    def or_else[**P, R](
        self,
        fn: Callable[Concatenate[E, P], Result[object, R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[T, R]:
        """Calls a function if the result is `Err`, otherwise returns the `Ok` value.

        This is often used for handling errors by trying an alternative operation.

        Args:
            fn (Callable[Concatenate[E, P], Result[object, R]]): The function to call with the `Err` value.
            *args (P.args): Additional positional arguments to pass to fn.
            **kwargs (P.kwargs): Additional keyword arguments to pass to fn.

        Returns:
            Result[T, R]: The original `Ok` value, or the result of the function if `Err`.

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

    def transpose[S](self: ResultType[Option[S], E]) -> Option[Result[S, E]]:
        """Transposes a Result containing an Option into an Option containing a Result.

        Can only be called if the inner type is `Option[S, E]`.

        The mapping is as follows:

        - `Ok(Some(v))` becomes `Some(Ok(v))`
        - `Ok(NONE)` becomes `NONE`
        - `Err(e)` becomes `Some(Err(e))`

        Returns:
            Option[Result[S, E]]: Option containing a Result or NONE.

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

    def or_[S, F](self, res: Result[S, F]) -> Result[T | S, F]:
        """Returns res if the result is `Err`, otherwise returns the `Ok` value of **self**.

        Args:
            res (Result[S, F]): The result to return if the original result is `Err`.

        Returns:
            Result[T | S, F]: The original `Ok` value, or `res` if the original result is `Err`.

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
    """

    value: Final[T]
    """Final[T]: The contained successful value."""

    __match_args__ = ("value",)
    # NOTE: this is an hack to avoid errors by immediatly casting `E` as `Any`, thus avoiding any type errors with incompatible types.
    @overload
    def __new__(cls, value: Result[T, E]) -> Result[Result[T, E], Any]: ...
    @overload
    def __new__(cls, value: Option[T]) -> Result[Option[T], Any]: ...
    @overload
    def __new__(cls, value: T) -> Result[T, Any]: ...

@final
class Err[T, E](ResultType[T, E]):
    """Represents an error value.

    One of the two variants of `Result[T, E]`, where `E` is the type of the value in `Err`.

    For more documentation, see the `ResultType[T, E]` Protocol.
    """

    error: Final[E]
    """Final[E]: The contained error value."""
    __match_args__ = ("error",)
    # NOTE: same hack as in `Ok` for type errors
    @overload
    def __new__(cls, error: Result[T, E]) -> Result[Any, Result[T, E]]: ...
    @overload
    def __new__(cls, error: Option[E]) -> Result[Any, Option[E]]: ...
    @overload
    def __new__(cls, error: E) -> Result[Any, E]: ...
