from __future__ import annotations

from abc import ABC
from collections.abc import Callable
from typing import Concatenate, Self


class Pipeable:
    def into[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Convert `Self` to `R`.

        This method allows to pipe the instance into an object or function that can convert `Self` into another type.

        Conceptually, this allow to do x.into(f) instead of f(x), hence keeping a functional chaining style.

        This is a core method, shared by all pyochain wrappers, that allows chaining operations in a functional style.

        Args:
            func (Callable[Concatenate[Self, P], R]): Function for conversion.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            R: The converted value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def maybe_sum(data: pc.Seq[int]) -> pc.Option[int]:
        ...     match data.length():
        ...         case 0:
        ...             return pc.NONE
        ...         case _:
        ...             return pc.Some(data.sum())
        >>>
        >>> pc.Seq(range(5)).into(maybe_sum).unwrap()
        10

        ```
        """
        return func(self, *args, **kwargs)

    def inspect[**P](
        self,
        func: Callable[Concatenate[Self, P], object],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Self:
        """Pass the instance to a function to perform side effects without altering the data.

        Args:
            func (Callable[Concatenate[Self, P], object]): Function to apply to the instance for side effects.
            *args (P.args): Positional arguments to pass to the function.
            **kwargs (P.kwargs): Keyword arguments to pass to the function.

        Returns:
            Self: The instance itself for chaining.

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


class CommonBase[T](ABC, Pipeable):
    """Base class for all wrappers.

    You can subclass this to create your own wrapper types.
    The pipe unwrap method must be implemented to allow piping functions that transform the underlying data type, whilst retaining the wrapper.

    Args:
        data (T): The underlying data to wrap.
    """

    _inner: T

    __slots__ = ("_inner",)

    def __init__(self, data: T) -> None:
        self._inner = data

    def inner(self) -> T:
        """Get the underlying data.

        This is a terminal operation that ends the chain.

        Returns:
            T: The underlying data.
        """
        return self._inner
