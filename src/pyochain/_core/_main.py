from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Concatenate, Self
from warnings import deprecated

from ._format import dict_repr

if TYPE_CHECKING:
    from .._dict import Dict
    from .._iter import Iter, Seq


class Pipeable:
    def pipe[**P, R](
        self,
        func: Callable[Concatenate[Self, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Pipe the instance in the function and return the result."""
        return func(self, *args, **kwargs)


class CommonBase[T](ABC, Pipeable):
    """
    Base class for all wrappers.
    You can subclass this to create your own wrapper types.
    The pipe unwrap method must be implemented to allow piping functions that transform the underlying data type, whilst retaining the wrapper.
    """

    _inner: T

    __slots__ = ("_inner",)

    def __init__(self, data: T) -> None:
        self._inner = data

    @abstractmethod
    def apply[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Any:
        raise NotImplementedError

    def println(self, pretty: bool = True) -> Self:
        """
        Print the underlying data and return self for chaining.

        Useful for debugging, simply insert `.println()` in the chain,
        and then removing it will not affect the rest of the chain.
        """
        from pprint import pprint

        if pretty:
            self.into(pprint, sort_dicts=False)
        else:
            self.into(print)
        return self

    def inner(self) -> T:
        """
        Return the underlying data.

        This is a terminal operation.
        """
        return self._inner

    @deprecated("Use .inner() instead")
    def unwrap(self) -> T:
        """
        Deprecated: Use `inner()` instead.
        """
        return self.inner()

    def into[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """
        Pass the *unwrapped* underlying data into a function.

        The result is not wrapped.
        ```python
        >>> import pyochain as pc
        >>> pc.Iter.from_(range(5)).into(list)
        [0, 1, 2, 3, 4]

        ```
        This is a core functionality that allows ending the chain whilst keeping the code style consistent.
        """
        return func(self.inner(), *args, **kwargs)

    def equals_to(self, other: Self | T) -> bool:
        """
        Check if two records are equal based on their data.

        Args:
            other: Another instance or corresponding underlying data to compare against.

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

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.inner().__repr__()})"

    def _eager[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], Sequence[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Seq[U]:
        from .._iter import Seq

        def _(data: Iterable[T]):
            return Seq(factory(data, *args, **kwargs))

        return self.into(_)

    def _lazy[**P, U](
        self,
        factory: Callable[Concatenate[Iterable[T], P], Iterator[U]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Iter[U]:
        from .._iter import Iter

        def _(data: Iterable[T]):
            return Iter(factory(data, *args, **kwargs))

        return self.into(_)


class MappingWrapper[K, V](CommonBase[dict[K, V]]):
    _inner: dict[K, V]

    def __repr__(self) -> str:
        return f"{self.into(dict_repr)}"

    def _new[KU, VU](self, func: Callable[[dict[K, V]], dict[KU, VU]]) -> Dict[KU, VU]:
        from .._dict import Dict

        return Dict(func(self.inner()))

    def apply[**P, KU, VU](
        self,
        func: Callable[Concatenate[dict[K, V], P], dict[KU, VU]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Dict[KU, VU]:
        """
        Apply a function to the underlying dict and return a Dict of the result.
        Allow to pass user defined functions that transform the dict while retaining the Dict wrapper.

        Args:
            func: Function to apply to the underlying dict.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        Example:
        ```python
        >>> import pyochain as pc
        >>> def invert_dict(d: dict[K, V]) -> dict[V, K]:
        ...     return {v: k for k, v in d.items()}
        >>> pc.Dict({'a': 1, 'b': 2}).apply(invert_dict)
        {1: 'a', 2: 'b'}

        ```
        """

        def _(data: dict[K, V]) -> dict[KU, VU]:
            return func(data, *args, **kwargs)

        return self._new(_)
