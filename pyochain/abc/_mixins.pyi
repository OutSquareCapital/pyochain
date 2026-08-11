from collections.abc import Callable
from typing import Concatenate, Protocol, Self

from pyochain import Option, Result

class Pipe(Protocol):
    """Mixin class providing the `pipe` method for fluent chaining."""
    def pipe[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Convert `Self` to `R`.

        This method allows to pipe the instance into an object or function that can convert `Self` into another type.

        Conceptually, this allow to do `x.pipe(f)` instead of `f(x)`, hence keeping a fluent chaining style.

        Args:
            func (Callable[Concatenate[Self, P], R]): Function for conversion.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            R: The converted value.

        Example:
            ```python
            from pyochain import Seq, Result, Ok, Err
            from collections.abc import Sequence

            def check_data(data: Sequence[int]) -> Result[Sequence[int], str]:
                if len(data) == 0:
                    return Err("Empty data")
                return Ok(data)

            def handle_result(res: Result[Sequence[int], str]) -> str:
                match res:
                    case Ok(data):
                        return f"Data is valid: {data}"
                    case Err(err):
                        return f"Data is invalid: {err}"

            x = Seq((1, 2, 3)).pipe(check_data).pipe(handle_result)
            assert x == "Data is valid: Seq(1, 2, 3)"
            ```
        """

class Tap(Protocol):
    """Mixin class providing the `tap` method."""
    def tap[**P](
        self,
        func: Callable[Concatenate[Self, P], object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Pass `Self` to **func**, call it, and return `Self` to continue chaining.

        This method is very useful for debugging or passing the instance to other functions for side effects (debugging, IO operations, logging, etc.), without breaking the fluent method chaining.

        The return type *assume* that **func** does not modify the instance, and that it returns `None` or any other value that is not used.

        Args:
            func (Callable[Concatenate[Self, P], object]): Function to apply to the instance for side effects.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            Self: The instance itself, unchanged.

        Example:
            ```python
            from pyochain import Seq, Vec

            v = Vec(())

            x = Seq((1, 2, 3, 4)).tap(v.extend).last()

            assert v == Vec((1, 2, 3, 4))
            assert x == 4
            ```
        """

class Fluent(Pipe, Tap, Protocol):
    """Mixin class providing `pipe` and `tap` methods for fluent chaining."""

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
        from pyochain import Seq, Some

        assert Seq((1, 2, 3)).then(sum) == Some(6)
        assert Seq(()).then(sum).is_none()
        ```
        This can also be extended to any type, not just collections.
        ```python
        from pyochain.abc import Checkable

        class MyString(str, Checkable): ...

        assert MyString("hello").then(lambda s: s.upper()) == Some("HELLO")
        assert MyString("").then(lambda s: s.upper()).is_none()
        ```
        This means that you can handle complex business logic in the same way.
        ```python
        from pyochain import Err
        from dataclasses import dataclass

        @dataclass(slots=True)
        class User(Checkable):
            name: str
            is_active: bool
            age: int
            def __bool__(self) -> bool:
                return self.is_active and self.age >= 18

            def describe(self) -> str:
                return f"{self.name} is an active adult"

        alice = User("Alice", is_active=True, age=30).then(User.describe)
        bob = (
            User("Bob", is_active=False, age=24)
            .then(User.describe)
            .ok_or("Expected an active adult user")
            .map_err(ValueError)
        )
        assert alice == Some("Alice is an active adult")
        assert (
            bob.map_err(repr).unwrap_err()
            == "ValueError('Expected an active adult user')"
        )
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
            from pyochain import Seq, Some

            assert Seq((1, 2, 3)).then(lambda s: s.iter().sum()) == Some(6)
            assert Seq(()).then(lambda s: s.iter().sum()).is_none()
            ```
        """

    def then_some(self) -> Option[Self]:
        """Wraps `Self` in an `Option[Self]` based on its truthiness.

        Returns:
            Option[Self]: `Some(self)` if self is truthy, `NONE` otherwise.

        Example:
            ```python
            from pyochain import Seq, Some

            data = Seq((1, 2, 3))

            assert data.then_some() == Some(data)
            assert Seq(()).then_some().is_none()
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
            from pyochain import Seq

            data = Seq((1, 2, 3))
            msg = "empty"

            assert data.ok_or(msg).unwrap() == data
            assert Seq(()).ok_or(msg).unwrap_err() == msg
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
            from pyochain import Seq

            msg = "should be empty"
            data = Seq((1, 2, 3))

            assert data.err_or(msg).unwrap_err() == data
            assert Seq(()).err_or(msg).unwrap() == msg
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
            from pyochain import Seq

            data = Seq((1, 2, 3))
            msg = "empty seq"

            assert data.ok_or_else(lambda s: msg).unwrap() == data
            assert Seq(()).ok_or_else(lambda s: msg).unwrap_err() == msg
            ```
        """
    def err_or_else[**P, T](
        self,
        func: Callable[Concatenate[Self, P], T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Result[T, Self]:
        """Wrap `Self` in a `Result[T, Self]` based on its truthiness.

        `T` being the return type of **func**.

        The function is only called if self evaluates to False.


        Args:
            func (Callable[Concatenate[Self, P], T]): A callable that returns the error value to wrap in Err.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Result[T, Self]: Ok(f(...)) if self is falsy, Err(self) otherwise.

        Example:
            ```python
            from pyochain import Seq

            msg = "should be empty"

            data = Seq((1, 2, 3))

            assert data.err_or_else(lambda s: msg).unwrap_err() == data
            assert Seq(()).err_or_else(lambda s: msg).unwrap() == msg
            ```
        """
