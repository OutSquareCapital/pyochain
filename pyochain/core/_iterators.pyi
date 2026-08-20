from collections.abc import Callable, Iterable, Iterator
from typing import Self, final, override

from pyochain import Option, Result
from pyochain.abc import PyoIterator

@final
class Iter[T](PyoIterator[T]):
    """Concrete implementation for `abc::PyoIterator`.

    Can be instantiated from any `Iterable` (like lists, sets, generators, etc.) efficiently (it only calls the builtin `iter()` on the input).

    As such, creating an `Iter` from an `Iterator` is virtually free.

    Tip:
        `Iter::__iter__()` returns the underlying wrapped `Iterator`, hence native speed is kept.

        i.e `Iter(...).map(f).collect(list)` is as fast as `list(map(f, [...]))`.

    See Also:
        [`abc::PyoIterator`][PyoIterator]: The abstract base class that `Iter` implements.

    Example:
        ```python
        from pyochain import Iter, Seq

        data = (0, 1, 2, 3, 4)

        assert Iter(data).collect(Seq) == Seq(data)
        iterator = Iter(data)

        # First we have a tuple iterator
        assert iterator.__iter__().__class__.__name__ == "tuple_iterator"

        # Now we have a map object
        mapped = iterator.map(lambda x: x * 2)
        assert mapped.__iter__().__class__.__name__ == "map"

        # We collect it, by default into a Seq
        assert mapped.collect(Seq) == Seq(0, 2, 4, 6, 8)

        # iterator is now exhausted
        assert iterator.collect(Seq) == Seq()
        ```
    """

    def __new__(cls, data: Iterable[T] | T = (), /, *more: T) -> Self:
        """Create a new `Iter` instance.

        If no arguments are provided, an empty `Iterator` is created.

        Args:
            data (Iterable[T] | T): Input data to create the `Iter` instance from.
            *more (T): Additional elements to yield from the iterator.

        Example:
            ```python
            from pyochain import Iter, Range

            data = (0, 1, 2, 3)

            # Create an `Iter` from an iterable
            assert Iter(data).collect(tuple) == Iter(Range(0, 4)).collect(tuple) == data

            # Create an `Iter` from individual elements
            assert Iter(0, 1, 2, 3).collect(tuple) == Iter(*data).collect(tuple) == data

            # Create an empty `Iter`
            assert 0 == Iter().count() == Iter(()).count() == Iter([]).count()
            ```
            You can also easily create an `Iter` from a generator expression:
            ```python
            from pyochain import Iter, Seq

            gen_expr = (x * x for x in range(5))
            assert Iter(gen_expr).collect(Seq) == Seq(0, 1, 4, 9, 16)
            ```
            Or from a generator function:
            ```python
            from pyochain import Iter

            def gen_func():
                for x in range(5):
                    yield x * x

            assert Iter(gen_func()).collect(Seq) == Seq(0, 1, 4, 9, 16)
            ```
        """
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
            from pyochain import Range, Some

            iterator = Range(0, 5).iter().peekable()

            # Peek at the first item of the iterator without consuming it.
            assert iterator.peek() == Some(0)

            # The next item returned is still 0, as we haven't consumed it yet.
            assert iterator.next() == Some(0)

            # Now the next item returned is 1, as we have consumed the first item.
            assert iterator.next() == Some(1)
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
            from pyochain import Range, Some

            iterator = Range(0, 5).iter().peekable()

            # The first item of the iterator is 0; consume it.
            assert iterator.next_if(lambda x: x == 0) == Some(0)

            # The next item returned is now 1, so `next_if` will return `None`.
            assert iterator.next_if(lambda x: x == 0).is_none()

            # `next_if` retains the next item if the predicate evaluates to `false` for it.
            assert iterator.next() == Some(1)
            ```
            Consume any number less than 10.
            ```python
            iterator = Range(1, 20).iter().peekable()

            # Consume all numbers less than 10
            while iterator.next_if(lambda x: x < 10).is_some():
                pass

            # The next value returned will be 10
            assert iterator.next() == Some(10)
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
            from pyochain import Range, Some

            iterator = Range(0, 5).iter().peekable()

            # The first item of the iterator is 0; consume it.
            assert iterator.next_if_eq(0) == Some(0)

            # The next item returned is now 1, so `next_if_eq` will return `None`.
            assert iterator.next_if_eq(0).is_none()

            # `next_if_eq` retains the next item if it was not equal to `expected`.
            assert iterator.next() == Some(1)
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
            from pyochain import Iter, Option, Some, NONE, Result
            import unicodedata

            iterator = Iter("125 GOTO 10").peekable()
            line_num = 0

            def try_parse_digit(c: str) -> Result[int, str]:
                try:
                    res = Some(unicodedata.digit(c))
                except ValueError as e:
                    res = NONE
                return res.ok_or(c)

            digit = iterator.next_if_map(try_parse_digit)
            while digit.is_some():
                line_num = line_num * 10 + digit.unwrap()
                digit = iterator.next_if_map(try_parse_digit)

            assert line_num == 125
            assert iterator.join("") == " GOTO 10"
            ```
        """
