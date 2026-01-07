"""Public mixins traits for internal pyochain types, and custom user implementations.

Since `Pipeable` and `Checkable` depend only on Self for arguments, returns types and internal logic, they can be safely added to any already existing class to provide additional functionality.

`PyoIterable` is a more specific trait, equivalent to subclassing `abc.Iterable`, but with additional methods and requirements specific to pyochain.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator
from typing import TYPE_CHECKING, Any, Concatenate, Self

import cytoolz as cz

if TYPE_CHECKING:
    from ._iter import Iter
    from ._option import Option
    from ._result import Result
__all__ = ["Checkable", "Pipeable", "PyoIterable"]


class Pipeable:
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
        >>> import hashlib
        >>> def sha256_hex(data: pc.Seq[int]) -> str:
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
        from ._option import NONE, Some

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
        from ._option import NONE, Some

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
        from ._result import Err, Ok

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
        from ._result import Err, Ok

        return Ok(self) if self else Err(func(self, *args, **kwargs))


class PyoIterable[I: Iterable[Any]](Pipeable, Checkable, Iterable[Any]):
    """Mixin trait class declaring that `Self` is an `Iterable`.

    This is equivalent to subclassing `abc.Iterable`, where you need to implement `__iter__`, but with the addition of the abstract method `iter` (which must return a pyochain `Iter`).

    **Signature and attributes**

    `PyoIterable` only contains the **_inner** attribute of type `I`, which must be an `Iterable` over `Any`.

    Due to limitation to Python generics, all subclass should be generic over a type `T` representing the element type.

    Otherwise, the custom methods implemented by the user in the subclass won't be able to infer the element type correctly.

    **Implementations needed**

    Since it also inerhit from `abc.Iterable`, `__iter__` must be declared, as well as the abstract method `iter`.

    The internal implementation shoud simply call, respectively:

    - `__iter__` -> `iter(self._inner)` (iter being the builtin function, not the abstract method defined below).
    - `iter` -> `Iter(self._inner)` (where `Iter` is the pyochain lazy iterator type).

    **Provided methods**

    `PyoIterable` also provide various methods for:
    - Comparison based on the underlying data
    - A generic `__repr__` implementation
    - A `length` method for counting elements (that can work on lazy sequences), equivalent to `len()`. Note that this doesn't implement `__len__` as that would break the `Iterable` protocol.
    - All methods from `Pipeable` and `Checkable`. Note that if the **_inner** is a lazy `Iterator`, methods from `Checkable` will always return `Ok`/`Some`, as the truthiness of an `Iterator` is always `True` in Python.

    Note:
        The comparison methods will consume any underlying `Iterator` instances involved in the comparison (for **self** and/or **other**).
        They are not implemented as dunder methods as a design choice, since Iterable types can be anything.
        The user should explicitly implement them if needed.

    Example:
    ```python
    >>> from pyochain import traits, Iter
    >>> from collections.abc import Iterable
    >>>
    >>> class MyIterable[T](traits.PyoIterable[list[T]]):
    ...     def __init__(self, data: Iterable[T]) -> None:
    ...         self._inner = list(data)
    ...
    ...     def __iter__(self) -> iter[T]:
    ...         return iter(self._inner)
    ...     def iter(self) -> Iter[T]:
    ...         return Iter(self._inner)
    >>>
    >>> my_iter = MyIterable([1, 2, 3])

    """

    _inner: I
    __slots__ = ("_inner",)
    __match_args__ = ("_inner",)

    @abstractmethod
    def iter(self) -> Iter[Any]:
        """Get an iterator over the `Iterable`.

        Call this to switch to lazy evaluation.

        Calling this method with an inner type who's a lazy `Iterator` instance has no effect.

        Returns:
            Iter[Any]: An `Iterator` over the `Iterable`. The element type is inferred from the actual subclass.
        """
        ...

    def __repr__(self) -> str:
        """Provides a generic representation for Iterable types.

        Should be overriden in subclasses that takes more specific _inner types, like polars.Series for example.
        """
        return f"{self.__class__.__name__}({_get_repr(self._inner).unwrap()})"

    def length(self) -> int:
        """Return the length of the Iterable.

        Like the builtin len but works on lazy sequences.

        Returns:
            int: The count of elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).length()
        2

        ```
        """
        return cz.itertoolz.count(self._inner)

    def eq(self, other: Self) -> bool:
        """Check if two Iterables are equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).eq(pc.Seq([1,2]))
        False
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2)))
        False
        >>> pc.Seq((1,2,3)).eq(pc.Vec([1,2,3]))
        True

        ```
        """
        return tuple(self._inner) == tuple(other._inner)

    def ne(self, other: Self) -> bool:
        """Check if two Iterables are not equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data are not equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2)))
        True
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) != tuple(other._inner)

    def le(self, other: Self) -> bool:
        """Check if this Iterable is less than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is less than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).le(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).le(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) <= tuple(other._inner)

    def lt(self, other: Self) -> bool:
        """Check if this Iterable is less than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is less than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).lt(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).lt(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) < tuple(other._inner)

    def gt(self, other: Self) -> bool:
        """Check if this Iterable is greater than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).gt(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).gt(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) > tuple(other._inner)

    def ge(self, other: Self) -> bool:
        """Check if this Iterable is greater than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).ge(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).ge(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) >= tuple(other._inner)


def _get_repr(data: Iterable[Any]) -> Result[str, str]:
    from pprint import pformat

    from ._result import Err, Ok

    def _repr_inner(data: Collection[Any]) -> Result[str, str]:
        return Ok(pformat(data, sort_dicts=False)[1:-1])

    match data:
        case Iterator():
            return Ok(data.__repr__())
        case Collection():
            match data:
                case set() | frozenset():
                    return _repr_inner(tuple(data))
                case _:
                    match len(data):
                        case 0:
                            return Ok("")
                        case _:
                            return _repr_inner(data)
        case _:
            return Err(
                f"Cannot provide generic representation for PyoIterable with _inner of type {type(data)}"
            )
