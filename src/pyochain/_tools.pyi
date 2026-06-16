from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Concatenate, Self, overload, override

from pyochain import Option, Result
from pyochain.abc import Position

from ._utils import no_doctest

@no_doctest
def try_find[T, E](
    data: Iterator[T], predicate: Callable[[T], Result[bool, E]]
) -> Result[Option[T], E]: ...
@no_doctest
def try_fold[T, B, E](
    data: Iterator[T], init: B, func: Callable[[B, T], Result[B, E]]
) -> Result[B, E]: ...
@no_doctest
def try_reduce[T, E](
    data: Iterator[T], func: Callable[[T, T], Result[T, E]]
) -> Result[Option[T], E]: ...
@no_doctest
def eq[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def ne[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def lt[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def gt[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def le[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def ge[T](data: Iterator[T], other: Iterable[T]) -> bool: ...
@no_doctest
def is_sorted[T](
    data: Iterator[T], *, reverse: bool = False, strict: bool = False
) -> bool: ...
@no_doctest
def is_sorted_by[T, U](
    data: Iterator[T],
    key: Callable[[T], U],
    *,
    reverse: bool = False,
    strict: bool = False,
) -> bool: ...
@no_doctest
def for_each[**P, T](
    data: Iterator[T],
    func: Callable[Concatenate[T, P], None],
    *args: P.args,
    **kwargs: P.kwargs,
) -> None: ...
@no_doctest
def for_each_star[**P, T](
    data: Iterator[T],
    func: Callable[..., Any],  # pyright: ignore[reportExplicitAny]
    *args: P.args,
    **kwargs: P.kwargs,
) -> None: ...
@no_doctest
def all_unique[T](data: Iterator[T]) -> bool: ...
@no_doctest
def all_unique_by[T, U](data: Iterator[T], key: Callable[[T], U]) -> bool: ...
@no_doctest
def partition[T](
    data: Iterator[T], predicate: Callable[[T], bool]
) -> tuple[list[T], list[T]]: ...
@no_doctest
def last[T](data: Iterator[T]) -> T: ...
@no_doctest
def count[T](data: Iterator[T]) -> int: ...
@no_doctest
def try_for_each[T, E](
    data: Iterator[T],
    f: Callable[[T], Result[Any, E]],  # pyright: ignore[reportExplicitAny]
) -> Result[tuple[()], E]: ...
@no_doctest
@overload
def try_collect[T](data: Iterator[Option[T]]) -> Option[list[T]]: ...
@overload
def try_collect[T, E](data: Iterator[Result[T, E]]) -> Option[list[T]]: ...
@no_doctest
def try_collect[T](
    data: Iterator[Option[T]] | Iterator[Result[T, Any]],  # pyright: ignore[reportExplicitAny]
) -> Option[list[T]]: ...
@no_doctest
def retain[T](data: MutableSequence[T], predicate: Callable[[T], bool]) -> None: ...
@no_doctest
def any[T](data: Iterator[T], predicate: Callable[[T], bool]) -> bool: ...  # noqa: A001
@no_doctest
def all[T](data: Iterator[T], predicate: Callable[[T], bool]) -> bool: ...  # noqa: A001

class Juxt:
    @overload
    def __init__(self, funcs: Iterable[Callable[..., object]], /) -> None: ...
    @overload
    def __init__(self, *funcs: Callable[..., object]) -> None: ...
    @no_doctest
    def __call__(self, *args: object) -> tuple[object, ...]: ...

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
