from collections.abc import Callable, Iterable, Iterator, MutableSequence
from typing import Any, Self, final, override

from ._utils import no_doctest
from .abc import PyoIterator
from .abc._iterator import Position
from .rs import Option, Result

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
    def __next__(self) -> T: ...

class UniqueKey[T](Iterator[T]):
    def __init__(self, data: Iterator[T], key: Callable[[T], object]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class Intersperse[T](Iterator[T]):
    def __init__(self, data: Iterator[T], element: T) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class SlidingWindow[T](Iterator[tuple[T, ...]]):
    def __init__(self, data: Iterator[T], n: int) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[T, ...]: ...

class FilterMap[T, R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[[T], Option[R]]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class FilterMapStar[T: Iterable[Any], R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[..., Option[R]]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class Scan[T, B](Iterator[B]):
    def __init__(
        self, data: Iterator[T], initial: B, func: Callable[[B, T], Option[B]]
    ) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> B: ...

class MapWhile[T, R](Iterator[R]):
    def __init__(self, data: Iterator[T], func: Callable[[T], Option[R]]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> R: ...

class FromFn[T](Iterator[T]):
    def __init__[**P](
        self, func: Callable[P, Option[T]], *args: P.args, **kwargs: P.kwargs
    ) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class Drain[T](Iterator[T]):
    def __init__(
        self, data: MutableSequence[T], start: int | None, end: int | None
    ) -> None: ...
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
    def __next__(self) -> T: ...

class Successors[T](Iterator[T]):
    def __init__(self, first: Option[T], succ: Callable[[T], Option[T]]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class FilterStar[T: Iterable[Any]](Iterator[T]):
    def __init__(self, data: Iterator[T], predicate: Callable[..., bool]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...

class WithPosition[T](Iterator[tuple[Position, T]]):
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[Position, T]: ...

class ZipLongest[T: Iterable[Any]](Iterator[tuple[Option[Any], ...]]):
    def __init__(self, data: Iterator[T]) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> tuple[Option[Any], ...]: ...

class Unzip[T](Iterator[T]):
    def __init__(self, data: Iterator[T], n: int) -> None: ...
    @no_doctest
    @override
    def __next__(self) -> T: ...
    @staticmethod
    def from_iterator[A, B](
        data: Iterator[tuple[A, B]],
    ) -> tuple[Unzip[A], Unzip[B]]: ...

@final
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
        >>> iterator.__iter__().__class__.__name__
        'tuple_iterator'
        >>> # Now we have a map object
        >>> mapped = iterator.map(lambda x: x * 2)
        >>> mapped.__iter__().__class__.__name__
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

    def __init__(self, data: Iterable[T]) -> None: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __next__(self) -> T: ...

@final
class Peekable[T](PyoIterator[T]):
    def __init__(self, iterable: Iterable[T]) -> None: ...
    @override
    def __iter__(self) -> Self: ...
    @override
    def __next__(self) -> T: ...
    def __bool__(self) -> bool: ...
    def peek(self) -> Option[T]:
        """Returns the `next()` value without advancing the `Iterator`.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if available, or `NONE` if the iteration is over.

        Examples:
            Peek at the next value of an iterator without consuming it.
            ```python
            >>> from pyochain import Range
            >>> iterator = Range(0, 5).iter().peekable()
            >>> # Peek at the first item of the iterator without consuming it.
            >>> iterator.peek()
            Some(0)
            >>> # The next item returned is still 0, as we haven't consumed it yet.
            >>> iterator.next()
            Some(0)
            >>> # Now the next item returned is 1, as we have consumed the first item.
            >>> iterator.next()
            Some(1)

            ```
        """
    def next_if(self, func: Callable[[T], bool]) -> Option[T]:
        """Consume and return the next value of this iterator if a condition is `True`.

        Args:
            func (Callable[[T], bool]): A function that takes the next value and returns a boolean.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if the condition is true, or `NONE` if the condition is false or the iteration is over.

        Examples:
            Consume a number if it's equal to 0.
            ```python
            >>> from pyochain import Range
            >>> iterator = Range(0, 5).iter().peekable()
            >>> # The first item of the iterator is 0; consume it.
            >>> iterator.next_if(lambda x: x == 0)
            Some(0)
            >>> # The next item returned is now 1, so `next_if` will return `None`.
            >>> iterator.next_if(lambda x: x == 0)
            NONE
            >>> # `next_if` retains the next item if the predicate evaluates to `false` for it.
            >>> iterator.next()
            Some(1)

            ```
            Consume any number less than 10.
            ```python
            >>> iterator = Range(1, 20).iter().peekable()
            >>> # Consume all numbers less than 10
            >>> while iterator.next_if(lambda x: x < 10).is_some():
            ...     pass
            >>> # The next value returned will be 10
            >>> iterator.next()
            Some(10)

            ```
        """
    def next_if_eq(self, expected: object) -> Option[T]:
        """Return the next item if it is equal to expected.

        Args:
            expected (object): The value to compare the next item against.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if it is equal to expected, or `NONE` if it is not equal or the iteration is over.

        Example:
            Consume a number if it's equal to 0.
            ```python
            >>> from pyochain import Range
            >>> iterator = Range(0, 5).iter().peekable()
            >>> # The first item of the iterator is 0; consume it.
            >>> iterator.next_if_eq(0)
            Some(0)
            >>> # The next item returned is now 1, so `next_if_eq` will return `None`.
            >>> iterator.next_if_eq(0)
            NONE
            >>> # `next_if_eq` retains the next item if it was not equal to `expected`.
            >>> iterator.next()
            Some(1)

            ```
        """

    def next_if_map[S, R](
        self: Peekable[S], f: Callable[[S], Result[R, S]]
    ) -> Option[R]:
        """Consumes the next value of this `Iterator` and applies a function *f* on it, returning the result if the closure returns `Ok`.

        Otherwise if the closure returns `Err` the value is put back for the next iteration.

        The content of the `Err` variant is typically the original value of the closure, but this is not required.

        If a different value is returned, the next `peek()` or `next()` call will result in this new value.

        Args:
            f (Callable[[S], Result[R, S]]): A function that takes the next value and returns a Result.

        Returns:
            Option[R]: The result of the function wrapped in `Some(R)` if the function returns `Ok(R)`, or `NONE` if the function returns `Err(S)` or the iteration is over.

        Examples:
            Parse the leading decimal number from an iterator of characters.
            ```python
            >>> from pyochain import Iter, Option, Some, NONE, Result
            >>> import unicodedata
            >>>
            >>> iterator = Iter("125 GOTO 10").peekable()
            >>> line_num = 0
            >>> def try_parse_digit(c: str) -> Result[int, str]:
            ...     try:
            ...         res = Some(unicodedata.digit(c))
            ...     except ValueError as e:
            ...         res = NONE
            ...     return res.ok_or(c)
            >>>
            >>> digit = iterator.next_if_map(try_parse_digit)
            >>> while digit.is_some():
            ...     line_num = line_num * 10 + digit.unwrap()
            ...     digit = iterator.next_if_map(try_parse_digit)
            >>> line_num
            125
            >>> iterator.join("")
            ' GOTO 10'

            ```
        """
