from __future__ import annotations

from collections.abc import Callable
from typing import Concatenate, Self

from ..rs import NONE, Err, Ok, Option, Result, Some


class Pipeable:
    """Mixin class providing pipeable methods for fluent chaining."""

    __slots__ = ()

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
        return func(self, *args, **kwargs)

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
        func(self, *args, **kwargs)
        return self


class Checkable:
    """Mixin class providing conditional chaining methods based on truthiness.

    This class provides methods inspired by Rust's `bool` type for conditional
    execution and wrapping in `Option` or `Result` types.

    All methods evaluate the instance's truthiness to determine their behavior.
    """

    __slots__ = ()

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
        return Some(func(self, *args, **kwargs)) if self else NONE

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
        return Some(self) if self else NONE

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
        return Ok(self) if self else Err(err)

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
        return Ok(self) if self else Err(func(self, *args, **kwargs))
