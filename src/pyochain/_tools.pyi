from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Self, override

from pyochain import Option

from ._abc import Position
from ._utils import no_doctest

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

class ZipLongest[T: Iterable[Any]](Iterator[tuple[Option[Any], ...]]):  # pyright: ignore[reportExplicitAny]
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __iter__(self) -> Self: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[Option[Any], ...]: ...  # pyright: ignore[reportExplicitAny]

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
