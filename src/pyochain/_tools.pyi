from collections.abc import Callable, Iterable, Iterator

from pyochain import Option, Result

from ._types import no_doctest

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
