from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from pprint import pformat
from typing import TYPE_CHECKING, Any, Concatenate, Self

if TYPE_CHECKING:
    from .._dict import Dict
    from .._iter import Iter, Seq


type IntoIter[T] = Iterator[T] | Generator[T, Any, Any]
"""A type alias representing an iterator or generator."""


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

    def tap(self, func: Callable[[Self], Any]) -> Self:
        """Tap into the chain to perform side effects without altering the data.

        Args:
            func (Callable[[Self], Any]): Function to apply to the instance for side effects.

        Returns:
            Self: The instance itself for chaining.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3, 4]).tap(print).last()
        Seq([1, 2, 3, 4])
        4

        ```
        """
        func(self)
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

    def equals_to(self, other: Self | T) -> bool:
        """Check if two records are equal based on their data.

        Args:
            other (Self | T): Another instance or corresponding underlying data to compare against.

        Returns:
            bool: True if the underlying data are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> d1 = pc.Dict({"a": 1, "b": 2})
        >>> d2 = pc.Dict({"a": 1, "b": 2})
        >>> d3 = pc.Dict({"a": 1, "b": 3})
        >>> d1.equals_to(d2)
        True
        >>> d1.equals_to(d3)
        False

        ```
        """
        other_data = other.inner() if isinstance(other, self.__class__) else other
        return self.inner() == other_data


class IterWrapper[T](CommonBase[Iterable[T]]):
    _inner: Iterable[T]

    def __iter__(self) -> Iterator[T]:
        return iter(self.inner())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inner().__repr__()})"

    def _eager[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], Sequence[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Seq[U]:
        from .._iter import Seq

        def _(data: Iterable[T]) -> Seq[U]:
            return Seq(factory(data, *args, **kwargs))

        return self.into(_)

    def _lazy[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], Iterator[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[U]:
        from .._iter import Iter

        def _(data: Iterable[T]) -> Iter[U]:
            return Iter(factory(data, *args, **kwargs))

        return self.into(_)


class MappingWrapper[K, V](CommonBase[dict[K, V]]):
    _inner: dict[K, V]

    def __repr__(self) -> str:
        def dict_repr(
            v: Mapping[Any, Any],
            max_items: int = 20,
            depth: int = 3,
            width: int = 80,
            *,
            compact: bool = True,
        ) -> str:
            truncated = dict(list(v.items())[:max_items])
            suffix = "..." if len(v) > max_items else ""
            return (
                pformat(truncated, depth=depth, width=width, compact=compact) + suffix
            )

        return f"{self.into(lambda d: dict_repr(d.inner()))}"

    def _new[KU, VU](self, func: Callable[[dict[K, V]], dict[KU, VU]]) -> Dict[KU, VU]:
        from .._dict import Dict

        return Dict(func(self.inner()))
