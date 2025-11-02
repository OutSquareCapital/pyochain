from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, Concatenate, Self

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

    _data: T

    __slots__ = ("_data",)

    def __init__(self, data: T) -> None:
        self._data = data

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

    def unwrap(self) -> T:
        """
        Return the underlying data.

        This is a terminal operation.
        """
        return self._data

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
        return func(self.unwrap(), *args, **kwargs)


class IterWrapper[T](CommonBase[Iterable[T]]):
    _data: Iterable[T]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.unwrap().__repr__()})"

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
    _data: dict[K, V]

    def _new[KU, VU](self, func: Callable[[dict[K, V]], dict[KU, VU]]) -> Dict[KU, VU]:
        from .._dict import Dict

        def _(data: dict[K, V]) -> Dict[KU, VU]:
            return Dict(func(data))

        return self.into(_)

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
        >>> pc.Dict({'a': 1, 'b': 2}).apply(invert_dict).unwrap()
        {1: 'a', 2: 'b'}

        ```
        """

        def _(data: dict[K, V]) -> dict[KU, VU]:
            return func(data, *args, **kwargs)

        return self._new(_)


class Wrapper[T](CommonBase[T]):
    """
    A generic Wrapper for any type.
    The pipe into method is implemented to return a Wrapper of the result type.

    This class is intended for use with other types/implementations that do not support the fluent/functional style.
    This allow the use of a consistent code style across the code base.
    """

    def apply[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Wrapper[R]:
        return Wrapper(self.into(func, *args, **kwargs))
