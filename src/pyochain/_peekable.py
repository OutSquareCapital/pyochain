from __future__ import annotations

from typing import TYPE_CHECKING, Final, Self, final, override

from .abc import PyoIterator
from .rs import NONE, Err, Null, Ok, Option, Result, Some, option

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator


@final
class Peekable[T](PyoIterator[T]):
    __slots__ = ("_iter", "_peeked")
    _iter: Final[Iterator[T]]
    _peeked: Option[T]

    def __init__(self, iterable: Iterable[T]) -> None:
        self._iter = iter(iterable)
        self._peeked = NONE

    @override
    def __iter__(self) -> Self:
        return self

    @override
    def __next__(self) -> T:
        match self._peeked:
            case Some(value):
                self._peeked = NONE
                return value
            case Null():
                return next(self._iter)

    def __bool__(self) -> bool:
        return self.peek().is_some()

    def peek(self) -> Option[T]:
        """Returns the `next()` value without advancing the `Iterator`.

        Returns:
            Option[T]: The next value wrapped in `Some(T)` if available, or `NONE` if the iteration is over.
        """
        match self._peeked:
            case Some(_):
                return self._peeked
            case Null():
                try:
                    self._peeked = option(next(self._iter))
                except StopIteration:
                    return NONE
                else:
                    return self._peeked

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
        match self.next():
            case Some(matched) if func(matched):
                return Some(matched)
            case other:
                self._peeked = other
                return NONE

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
        return self.next_if(lambda nxt: nxt == expected)

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
            >>> from pyochain import Iter, Option, Some, NONE
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
        match self.next():
            case Some(item):
                match f(item):
                    case Ok(result):
                        return Some(result)
                    case Err(item):
                        unpeek = Some(item)
            case Null():
                unpeek = NONE

        self._peeked = unpeek
        return NONE
