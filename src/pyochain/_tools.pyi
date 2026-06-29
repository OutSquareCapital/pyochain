from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Final, Self, override

from pyochain import Option
from pyochain.abc import PyoIterator

from ._utils import no_doctest
from .abc._iterator import Position

@no_doctest
def retain[T](data: MutableSequence[T], predicate: Callable[[T], bool]) -> None: ...

class MapJuxt[R](Iterator[tuple[R, ...]]):
    def __init__(
        self, iterator: Iterator[object], *funcs: Callable[..., R]
    ) -> None: ...
    @no_doctest
    def __next__(self) -> tuple[R, ...]: ...

class UniqueIdentity[T](Iterator[T]):
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class UniqueKey[T](Iterator[T]):
    def __init__(self, data: Iterator[T], key: Callable[[T], object]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class Intersperse[T](Iterator[T]):
    def __init__(self, data: Iterator[T], element: T) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class SlidingWindow[T](Iterator[tuple[T, ...]]):
    def __init__(self, data: Iterator[T], n: int) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[T, ...]: ...

class FilterMap[T, R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[[T], Option[R]]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class FilterMapStar[T: Iterable[Any], R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[..., Option[R]]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class Scan[T, B](Iterator[B]):
    def __init__(
        self, data: Iterator[T], initial: B, func: Callable[[B, T], Option[B]]
    ) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> B: ...

class MapWhile[T, R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[[T], Option[R]]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class FromFn[T](Iterator[T]):
    def __init__[**P](
        self, func: Callable[P, Option[T]], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class Drain[T](Iterator[T]):
    def __init__(
        self, data: MutableSequence[T], start: int | None, end: int | None
    ) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class ExtractIf[T](Iterator[T]):
    def __init__(
        self,
        data: MutableSequence[T],
        predicate: Callable[[T], bool],
        start: int,
        end: int | None,
    ) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class Successors[T](Iterator[T]):
    def __init__(self, first: Option[T], succ: Callable[[T], Option[T]]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class FilterStar[T: Iterable[Any]](Iterator[T]):
    def __init__(self, data: Iterator[T], predicate: Callable[..., bool]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class WithPosition[T](Iterator[tuple[Position, T]]):
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[Position, T]: ...

class ZipLongest[T: Iterable[Any]](Iterator[tuple[Option[Any], ...]]):
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[Option[Any], ...]: ...

class Unzip[T](Iterator[T]):
    def __init__(self, data: Iterator[T], n: int) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...
    @staticmethod
    def from_iterator[A, B](
        data: Iterator[tuple[A, B]],
    ) -> tuple[Unzip[A], Unzip[B]]: ...

class Iter[T](PyoIterator[T]):
    """Concrete implementation for `abc::PyoIterator`.

    Can be instantiated from any `Iterable` (like lists, sets, generators, etc.) efficiently (it only calls the builtin `iter()` on the input).

    As such, creating an `Iter` from an `Iterator` is virtually free.

    Tip:
        `Iter::__iter__()` returns the underlying wrapped `Iterator`, hence native speed is kept.

        i.e `Iter([...]).map(f).collect(list)` is as fast as `list(map(f, [...]))`.

    Args:
        data (Iterable[T]): Any object that can be iterated over.

    See Also:
        [`abc::PyoIterator`][PyoIterator]: The abstract base class that `Iter` implements.

    Example:
        ```python
        >>> from pyochain import Iter, Seq
        >>>
        >>> data = (0, 1, 2, 3, 4)
        >>> Iter(data).collect(Seq)
        Seq(0, 1, 2, 3, 4)
        >>> iterator = Iter(data)
        >>> # First we have a tuple iterator
        >>> iterator._inner.__class__.__name__
        'tuple_iterator'
        >>> # Now we have a map object
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped._inner.__class__.__name__
        'map'
        >>> # We collect it, by default into a Seq
        >>> mapped.collect(Seq)
        Seq(0, 2, 4, 6, 8)
        >>> # iterator is now exhausted
        >>> iterator.collect(Seq)
        Seq()

        ```
        You can also easily create an `Iter` from a generator expression:
        ```python
        >>> from pyochain import Iter
        >>> gen_expr = (x * x for x in range(5))
        >>> Iter(gen_expr).collect(Seq)
        Seq(0, 1, 4, 9, 16)

        ```
        Or from a generator function:
        ```python
        >>> from pyochain import Iter
        >>> def gen_func():
        ...     for x in range(5):
        ...         yield x * x
        >>>
        >>> Iter(gen_func()).collect(Seq)
        Seq(0, 1, 4, 9, 16)

        ```
    """

    _inner: Final[Iterator[T]]

    def __init__(self, data: Iterable[T]) -> None: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __next__(self) -> T: ...
