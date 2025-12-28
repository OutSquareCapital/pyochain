from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Concatenate, Protocol, Self


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


# typeshed protocols


class SupportsDunderLT[T](Protocol):
    def __lt__(self, other: T, /) -> bool: ...


class SupportsDunderGT[T](Protocol):
    def __gt__(self, other: T, /) -> bool: ...


class SupportsDunderLE[T](Protocol):
    def __le__(self, other: T, /) -> bool: ...


class SupportsDunderGE[T](Protocol):
    def __ge__(self, other: T, /) -> bool: ...


class SupportsKeysAndGetItem[K, V](Protocol):
    def keys(self) -> Iterable[K]: ...
    def __getitem__(self, key: K, /) -> V: ...


type SupportsRichComparison[T] = SupportsDunderLT[T] | SupportsDunderGT[T]
