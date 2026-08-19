from abc import abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, MutableSequence
from typing import (
    Any,
    Concatenate,
    Literal,
    Protocol,
    TypeGuard,
    TypeIs,
    overload,
    override,
    runtime_checkable,
)

from _typeshed import SupportsRichComparison

from pyochain import Option, Peekable, Result, Vec
from pyochain.abc import PyoIterable, PyoMutableSequence

from .._types import (
    LiteralInteger,
    SupportsAnyAdd,
    SupportsComparison,
    SupportsSumWithNoDefaultGiven,
)

type Position = Literal["first", "middle", "last", "only"]
"""Type representing the position of an item in an `Iterator`."""

type AnyOpt = Option[Any]
type ZippedLongest[T] = (
    PyoIterator[tuple[Option[T], AnyOpt]]
    | PyoIterator[tuple[Option[T], AnyOpt, AnyOpt]]
    | PyoIterator[tuple[Option[T], AnyOpt, AnyOpt, AnyOpt]]
    | PyoIterator[tuple[Option[T], AnyOpt, AnyOpt, AnyOpt, AnyOpt]]
    | PyoIterator[tuple[AnyOpt, ...]]
)
"""Type representing the result of a `zip_longest` operation, which can yield tuples of varying lengths depending on the number of iterables zipped together."""

type FilterFn[T, R] = Callable[[T], object | TypeIs[R] | TypeGuard[R]] | None
"""Optional closure that can be passed to `PyoIterator::filter` to determine if an element should be yielded."""

@runtime_checkable
class PyoIterator[T](PyoIterable[T], Protocol):
    """Extends `PyoIterable[T]` and `collections.abc.Iterator[T]`.

    - An `Iterable` is any object capable of creating an `Iterator` (i.e., it implements the `__iter__()` method).
    - An `Iterator` is an object representing a stream of data, generating the next value with each call to `__next__()`.

    `Iterator`s are composable, meaning you can chain operations like `map()`, `filter()`, etc., that will simply add a new step to the processing pipeline without executing it.

    Thus, it can be considered akin to a SQL query: An `Iterator` represents a recipe for how to process the data.

    Terminal operations (like `collect()`, `count()`, `all()`, etc.) will "execute the query" by consuming the `Iterator` and producing a final result.

    This is done by calling `__next__()` repeatedly until `StopIteration` is raised, which signals that the `Iterator` is exhausted.

    Once this happened, the `Iterator` instance is empty and cannot be reused to produce new values.

    A high-level way of thinking about how to use `Iterators` is to create one from a source of data, build a plan, and execute it.

    Then, if the result is a new `Iterable`, you can create a new `Iterator` from it and repeat the process.

    If all of this doesn't sound familiar, it's simply because Python does this in an implicit way.

    A *for loop* will create an `Iterator` from the provided iterable, and consume it until exhaustion.

    For example, a `list` knows its size, how to access items by index, etc..

    But it does not know how to iterate over itself, i.e returns elements one by one and stop once x event happens.

    It knows, however, how to create an `Iterator` object that will handle this.

    All concrete subclasses must implement the required `Iterator` dunder methods:

    - `__iter__`
    - `__next__`

    Example:
        ```python
        from pyochain import Seq, Some
        from pyochain.abc import PyoIterator

        class Count(PyoIterator[int]):
            def __init__(self, start: int = 0):
                self.current = start

            def __iter__(self):
                return self

            def __next__(self):
                val = self.current
                self.current += 1
                return val

        counter = Count(5)
        assert counter.next() == Some(5)
        assert counter.next() == Some(6)
        assert counter.iter().take(3).collect(Seq) == Seq(7, 8, 9)
        ```
    """
    @abstractmethod
    def __next__(self) -> T: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @classmethod
    def from_count(cls, start: int = 0, step: int = 1) -> PyoIterator[int]:
        """Create an `Iterator` of evenly spaced values, beginning with *start*.

        Can be used with `map()` to generate consecutive data points or with `zip()` to add sequence numbers.

        Warning:
            The `Iterator` returned is **infinite**, meaning it will never stop yielding elements.

            Be sure to use `PyoIterator::take` or `PyoIterator::slice` to limit the number of items taken.

            Otherwise you could quickly run out of memory, if you try to collect it into a collection.

        Args:
            start (int): Starting value of the `Iterator`.
            step (int): Difference between consecutive values.

        Returns:
            PyoIterator[int]: An `Iterator` of integers starting from **start** and increasing by **step**.

        Example:
            ```python
            from pyochain import Iter, Seq

            assert Iter.from_count(10, 2).take(3).collect(Seq) == Seq(10, 12, 14)
            assert Iter.from_count(-5, 5).take(4).collect(Seq) == Seq(-5, 0, 5, 10)
            assert Iter.from_count(0, -1).take(5).collect(Seq) == (0, -1, -2, -3, -4)
            x = Iter.from_count(0, 5).map(lambda x: x**2).take(4).collect(tuple)
            assert x == (0, 25, 100, 225)
            y = Iter.from_count(0, 5).zip([1, 2, 3]).collect(tuple)
            assert y == ((0, 1), (5, 2), (10, 3))
            ```
        """

    @classmethod
    def from_fn[**P, R](
        cls, f: Callable[P, Option[R]], *args: P.args, **kwargs: P.kwargs
    ) -> PyoIterator[R]:
        r"""Create an `Iterator` from a generator function.

        The `Callable` must return:

        - `Some(value)` to yield a value
        - `NONE` to stop the iteration

        You could consider this as a way to create an `Iterator` where the `__next__()` is the `__call__()` method.

        As such, you can either provide lambdas, partials, closures, or pre-existing classes where `__call__()` is implemented, but a `__next__()` is not desired.

        If you do have an `Iterator` class, simply pass it to the regular constructor, as this will be more efficient, ergonomic and idiomatic.

        Args:
            f (Callable[P, Option[R]]): `Callable` that returns the next item wrapped in `Option`.
            *args (P.args): Positional arguments to pass to **f**.
            **kwargs (P.kwargs): Keyword arguments to pass to **f**.

        Returns:
            PyoIterator[R]: An `Iterator` yielding values produced by **f**.

        Note:
            In Rust, this avoids defining a full struct and implementing `Iterator` for it when you have simple logic to generate values.

            This is implemented for "Rust API compliance", but in Python, generators comprehensions/functions with `yield` statements are the ergonomic equivalent.

        Example:
            Closure with captured local variable:
            ```python
            from pyochain import Iter, Some, NONE, Option

            def make_counter(max_val: int):
                counter = 0

                def gen() -> Option[int]:
                    nonlocal counter
                    counter += 1
                    return Some(counter) if counter <= max_val else NONE

                return gen

            x = Iter.from_fn(make_counter(5)).collect(tuple)
            assert x == (1, 2, 3, 4, 5)
            ```
            Reading records from a text stream:
            ```python
            from io import StringIO

            stream = StringIO("Alice\nBob\nCharlie\n")

            def read_name() -> Option[str]:
                line = stream.readline()
                return Some(line.rstrip("\n")) if line else NONE

            iterator = Iter.from_fn(read_name)
            assert iterator.next() == Some("Alice")
            assert iterator.next() == Some("Bob")
            assert iterator.next() == Some("Charlie")
            assert iterator.next().is_none()
            ```
        """

    @classmethod
    def once[V](cls, value: V) -> PyoIterator[V]:
        """Create an `Iterator` that yields a single value.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.


        Args:
            value (V): The single value to yield.

        Returns:
            PyoIterator[V]: An `Iterator` yielding the specified value.

        Example:
            ```python
            from pyochain import Iter, Seq

            assert Iter.once(42).collect(Seq) == Seq(
                42,
            )
            ```
        """

    @classmethod
    def once_with[**P, R](
        cls, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> PyoIterator[R]:
        """Create an `Iterator`  that lazily generates a value exactly once by invoking the provided closure.

        If you have a function which works on iterators, but you only need to process one value, you can use this method rather than doing something like `Iter([value])`.

        This can be considered the lazy counterpart of [`PyoIterator::once`][PyoIterator.once].

        Unlike `PyoIterator::once`, this function will lazily generate the value on request.

        Args:
            func (Callable[P, R]): The single value to yield.
            *args (P.args): Positional arguments to pass to **func**.
            **kwargs (P.kwargs): Keyword arguments to pass to **func**.

        Returns:
            PyoIterator[R]: An `Iterator` yielding the specified value.

        Example:
            ```python
            from pyochain import Iter, Seq

            assert Iter.once_with(lambda: 42).collect(Seq) == Seq(
                42,
            )
            ```
        """

    @classmethod
    def repeat[O](cls, obj: O, n: int | None = None) -> PyoIterator[O]:
        """Repeat the provided object **n** times as elements of an `Iterator`.

        If **n** is `None`, this will create an infinite `Iterator`.

        Be sure to use [`PyoIterator::take`][PyoIterator.take] or [`PyoIterator::slice`][PyoIterator.slice] to limit the number of items taken.

        Warning:
            Each repetition is a reference to the same object, not a copy.

            This means that if the object is mutable and you modify one of the repetitions, all next repetitions will reflect that change.

        Args:
            obj (O): The object to repeat.
            n (int | None): Optional number of repetitions.

        Returns:
            PyoIterator[O]: An `Iterator` of repeated **obj**.

        See Also:
            [`PyoIterator::cycle`][cycle] to repeat the **elements** of the `Iterator`.

        Example:
            ```python
            from pyochain import Seq, Iter

            assert Iter.repeat(1, 3).collect(Seq) == (1, 1, 1)
            assert Iter.repeat(("a", "b"), 2).collect(Seq) == (("a", "b"), ("a", "b"))
            ```
            A common use for repeat is to supply a stream of constant values to map or zip:
            ```python
            from pyochain import Range

            out = Range(0, 10).iter().map_with(pow, Iter.repeat(2)).collect(Seq)
            assert out == (0, 1, 4, 9, 16, 25, 36, 49, 64, 81)
            ```

            Shared reference behavior:
            ```python
            from pyochain import Vec

            base = ["Alice", "Bob", "Charlie"]

            first, second = Iter.repeat(base).take(2).collect(tuple)
            first.append("Joe")

            assert first == ["Alice", "Bob", "Charlie", "Joe"]
            assert base == ["Alice", "Bob", "Charlie", "Joe"]
            assert second == ["Alice", "Bob", "Charlie", "Joe"]
            assert first is second and first is base and second is base
            ```
        """

    @classmethod
    def successors[U](
        cls, first: Option[U], succ: Callable[[U], Option[U]]
    ) -> PyoIterator[U]:
        """Create an iterator of successive values computed from the previous one.

        The iterator yields `first` (if it is `Some`), then repeatedly applies **succ** to the
        previous yielded value until it returns `NONE`.

        Args:
            first (Option[U]): Initial item.
            succ (Callable[[U], Option[U]]): Successor function.

        Returns:
            PyoIterator[U]: `Iterator` yielding `first` and its successors.

        Example:
            ```python
            from pyochain import Iter, Some, NONE, Option, Seq

            def next_pow10(x: int) -> Option[int]:
                return Some(x * 10) if x < 10_000 else NONE

            a = Iter.successors(Some(1), next_pow10).collect(Seq)
            assert a == (1, 10, 100, 1000, 10000)
            b = Iter.successors(NONE, next_pow10).collect(Seq)
            assert b == ()
            ```
        """

    @overload
    def accumulate[S](
        self: PyoIterator[S], func: None = None, initial: S | None = None
    ) -> PyoIterator[S]: ...
    @overload
    def accumulate[I, N](
        self: PyoIterator[N],
        func: Callable[[I, N], I],
        initial: I | None = None,
    ) -> PyoIterator[I]: ...
    def accumulate[S](
        self: PyoIterator[S],
        func: Callable[[S, S], S] | None = None,
        initial: S | None = None,
    ) -> PyoIterator[S]:
        """Return an `Iterator` of accumulated binary **function** results.

        In principle, `PyoIterator::accumulate` is similar to `PyoIterator::fold` if you provide it with the same binary function.

        However, instead of returning the final accumulated result, it returns an `Iterator` that yields the current value `T` of the accumulator for each iteration.

        In other words, the last element yielded by `PyoIterator::accumulate` is what would have been returned by `PyoIterator::fold` if it had been used instead.

        **function** should accept two arguments, an accumulated total and a value from the `Iterator`.

        Args:
            func (Callable[[S, S], S] | None): Optional binary function to apply cumulatively. If `None`, the default is to use addition (`operator.add`).
            initial (S | None): Optional initial value to start the accumulation.

        Returns:
            PyoIterator[S]: A new `Iterator` with accumulated results.

        Example:
            ```python
            import operator as op
            from pyochain import Seq

            s = Seq(1, 2, 3)
            assert s.iter().accumulate().collect(tuple) == (1, 3, 6)
            assert s.iter().accumulate(initial=10).collect(tuple) == (10, 11, 13, 16)
            assert s.iter().accumulate(op.mul).collect(tuple) == (1, 2, 6)
            assert s.iter().accumulate(op.add, 0).collect(Seq) == (0, 1, 3, 6)
            # The final accumulated result is the same as fold:
            assert s.iter().fold(0, op.add) == 6
            assert s.iter().accumulate(op.mul).collect(Seq) == (1, 2, 6)
            assert s.iter().accumulate().collect(Seq) == (1, 3, 6)
            ```
            To compute a running minimum, set function to `min()`.

            For a running maximum, set function to `max()`.

            Or for a running product, set function to `operator.mul()`.

            To build an amortization table, accumulate the interest and apply payments:
            ```python
            from pyochain import Iter
            import operator

            data = Seq(3, 4, 6, 2, 1, 9, 0, 7, 5, 8)
            running_max = data.iter().accumulate(max).collect(Seq)
            assert running_max == (3, 4, 6, 6, 6, 9, 9, 9, 9, 9)

            running_product = data.iter().accumulate(operator.mul).collect(Seq)
            assert running_product == (3, 12, 72, 144, 144, 1296, 0, 0, 0, 0)

            # Amortize a 5% loan of 1000 with 10 annual payments of 90
            update = lambda balance, payment: round(balance * 1.05) - payment
            res = Iter.repeat(90, 10).accumulate(update, initial=1_000).collect(list)
            assert res == [1000, 960, 918, 874, 828, 779, 728, 674, 618, 559, 497]
            ```
        """

    def all(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if every element of the `Iterator` is truthy.

        `PyoIterator::.all` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterator`, and if they all return true, then so does `PyoIterator::.all`.

        If any of them return false, it returns false.

        An empty `Iterator` returns true.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if all elements match the predicate, False otherwise.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(1, True).iter().all()
            assert Seq().iter().all()
            assert not Seq(1, 0).iter().all()

            def is_even(x: int) -> bool:
                return x % 2 == 0

            assert Seq(2, 4, 6).iter().all(is_even)
            assert not Seq("a", "", "c").iter().all()
            assert not Seq(1, None, 3).iter().all()
            ```
        """

    def all_equal[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Return `True` if all items of the `Iterator` are equal.

        A function that accepts a single argument and returns a transformed version of each input item can be specified with **key**.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison.

        Returns:
            bool: `True` if all items are equal, `False` otherwise.

        Example:
            ```python
            from pyochain import Seq, Range

            assert Seq("AaaA").iter().all_equal(key=str.casefold)
            assert Range(0, 9).iter().all_equal(key=lambda x: x < 10)
            assert not Range(0, 9).iter().all_equal()
            ```
        """

    def all_unique[U](self) -> bool:
        """Returns `True` if all the elements of the `Iterator` are unique.

        The function returns as soon as the first non-unique element is encountered.

        Elements are assumed to be hashable.

        If you need to check uniqueness based on a custom key function, use `PyoIterable::all_unique_by` instead.

        Tip:
            If you already have an existing `Collection`, you can alternatively check uniqueness by comparing the length of the collection to the length of a set created from it.

            On a "worst" case scenario (all elements are unique), this can be a bit faster on large (100k + items) collections, by around 1.15x (i.e 15% faster).

            Or on very small (10 items or less), where the overhead of creating the `Iterator` makes it 2x slower than simply creating the set.

            Altough, at this point, the operation is so fast that the difference is negligible, unless you are doing it in a hot loop.

            All things considered, `all_unique` early-exits on first duplicate can make it orders of magnitude faster, when your probability of duplicates is anything but very low.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
            ```python
            from pyochain import Seq, Set

            assert not Seq("ABCB").iter().all_unique()
            assert Seq("ABCb").iter().all_unique()

            # Alternative way to check uniqueness by comparing lengths:
            collection = Seq(1, 2, 3, 3)
            assert not collection.len() == collection.pipe(Set).len()
            ```
        """

    def all_unique_by[U](self, key: Callable[[T], U]) -> bool:
        """Returns True if all the elements of **self** transformed by **key** are unique.

        The function returns as soon as the first non-unique element is encountered.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], U]): Function to transform items before comparison.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
            ```python
            from pyochain import Seq

            assert Seq("ABCb").iter().all_unique()
            assert not Seq("ABCb").iter().all_unique_by(str.lower)
            ```
        """

    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if any element of the `Iterator` is truthy.

        `PyoIterator::.any` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterator`, and if any of them return true, then so does `PyoIterator::.any`.

        If they all return false, it returns false.

        An empty `Iterator` returns false.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if any element matches the predicate, False otherwise.

        Example:
            ```python
            from pyochain import Seq, Range

            assert Seq(0, 1).iter().any()
            assert not Range(0, 0).iter().any()

            def is_even(x: int) -> bool:
                return x % 2 == 0

            assert Seq(1, 3, 4).iter().any(is_even)
            ```
        """

    def arg_max(self) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator`.

        Credits to more-itertools for the implementation.

        Returns:
            int: The index of the maximum value.

        Example:
            Basic usage:
            ```python
            from pyochain import Iter, Seq

            assert Iter("abcdefghabcd").arg_max() == 7
            assert Iter((0, 1, 2, 3, 3, 2, 1, 0)).arg_max() == 3
            ```
            Identify the best machine learning model:
            ```python
            models = Seq("svm", "random forest", "knn", "naïve bayes")
            accuracy = Seq(68, 61, 84, 72)

            # Most accurate model
            assert models.get(accuracy.iter().arg_max()).unwrap() == "knn"

            # Best accuracy
            assert accuracy.iter().max() == 84
            ```
        """

    def arg_max_by[U](self, key: Callable[[T], U]) -> int:
        """Index of the first occurrence of a maximum value in the `Iterator` based on a *key* function.

        The *key* function must accept a single argument and return a transformed, comparable version of each input item.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U]): Function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        Example:
            Basic usage:
            ```python
            from pyochain import Seq

            assert Seq("a", "bbb", "cc").iter().arg_max_by(len) == 1
            assert Seq("Alice", "bob", "charlie").iter().arg_max_by(str.lower) == 2
            ```
            Identify the best machine learning model:
            ```python
            models = Seq("svm", "random forest", "knn", "naïve bayes")
            accuracy = Seq("68", "61", "84", "72")

            # Most accurate model
            assert models.get(accuracy.iter().arg_max_by(int)).unwrap() == "knn"

            # Best accuracy
            assert accuracy.iter().max_by(int) == "84"
            ```
        """

    def arg_min(self) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator`.

        Credits to **more-itertools** for the examples.

        Returns:
            int: The index of the minimum value.

        Example:
            ```python
            from pyochain import Seq

            assert Seq("efghabcdijkl").iter().arg_min() == 4
            assert Seq(3, 2, 1, 0, 4, 2, 1, 0).iter().arg_min() == 3
            ```
        """

    def arg_min_by[U](self, key: Callable[[T], U]) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator` based on a *key* function.

        The *key* function must accept a single argument and return a transformed, comparable version of each input item.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U]): Function to determine the value for comparison.

        Returns:
            int: The index of the minimum value.

        Example:
            Basic usage:
            ```python
            from pyochain import Seq

            assert Seq("aaa", "b", "cc").iter().arg_min_by(len) == 1
            assert Seq("Alice", "bob", "Charlie").iter().arg_min_by(str.lower) == 0
            ```
            Find the fastest healing family member based on age:
            ```python
            def cost(x: int) -> float:
                "Days for a wound to heal given a subject's age."
                return x**2 - 20 * x + 150

            labels = Seq("homer", "marge", "bart", "lisa", "maggie")
            ages = Seq(35, 30, 10, 9, 1)
            # Fastest healing family member
            assert labels.get(ages.iter().arg_min_by(cost)).unwrap() == "bart"
            # Age with fastest healing
            assert ages.iter().min_by(key=cost) == 10
            ```
        """

    @overload
    def batched(
        self, n: Literal[1], *, strict: Literal[True]
    ) -> PyoIterator[tuple[T]]: ...
    @overload
    def batched(
        self, n: Literal[2], *, strict: Literal[True]
    ) -> PyoIterator[tuple[T, T]]: ...
    @overload
    def batched(
        self, n: Literal[3], *, strict: Literal[True]
    ) -> PyoIterator[tuple[T, T, T]]: ...
    @overload
    def batched(
        self, n: Literal[4], *, strict: Literal[True]
    ) -> PyoIterator[tuple[T, T, T, T]]: ...
    @overload
    def batched(
        self, n: Literal[5], *, strict: Literal[True]
    ) -> PyoIterator[tuple[T, T, T, T, T]]: ...
    @overload
    def batched(
        self, n: int, *, strict: Literal[False]
    ) -> PyoIterator[tuple[T, ...]]: ...
    @overload
    def batched(
        self, n: int, *, strict: bool = False
    ) -> PyoIterator[tuple[T, ...]]: ...
    def batched(self, n: int, *, strict: bool = False) -> PyoIterator[tuple[T, ...]]:
        """Batch elements into tuples of length n and return a new `Iterator`.

        - The last batch may be shorter than n.
        - The data is consumed lazily, just enough to fill a batch.
        - The result is yielded as soon as a batch is full or when the `Iterator` is exhausted.

        Note:
            This is the closest equivalent to `Iterator::array_chunks` in Rust.

        Args:
            n (int): Number of elements in each batch.
            strict (bool): If `True`, raises a ValueError if the last batch is not of length n.

        Returns:
            PyoIterator[tuple[T, ...]]: An iterable of batched tuples.

        Example:
            ```python
            from pyochain import Seq

            a = Seq("ABCDEFG").iter().batched(3).collect(Seq)
            b = (("A", "B", "C"), ("D", "E", "F"), ("G",))
            assert a == b
            data = Seq(1, 1, 2, -2, 6, 0, 3, 1, 0)
            #           ^-----^  ^------^  ^-----^
            assert data.iter().batched(3, strict=True).map(sum).all(lambda x: x == 4)
            ```
            You can use it to group elements into fixed-size groups.
            ```python
            from pyochain import Vec

            flattened_data = Vec("roses", "red", "violets", "blue", "sugar", "sweet")
            unflattened = flattened_data.iter().batched(2).collect(Vec)
            assert unflattened == [
                ("roses", "red"),
                ("violets", "blue"),
                ("sugar", "sweet"),
            ]
            ```
        """

    @overload
    def chain[S, O1](
        self: PyoIterator[S], o1: Iterable[O1], /
    ) -> PyoIterator[S | O1]: ...
    @overload
    def chain[S, O1, O2](
        self: PyoIterator[S], o1: Iterable[O1], o2: Iterable[O2], /
    ) -> PyoIterator[S | O1 | O2]: ...
    @overload
    def chain[S, O1, O2, O3](
        self: PyoIterator[S], o1: Iterable[O1], o2: Iterable[O2], o3: Iterable[O3], /
    ) -> PyoIterator[S | O1 | O2 | O3]: ...
    @overload
    def chain[S, O1, O2, O3, O4](
        self: PyoIterator[S],
        o1: Iterable[O1],
        o2: Iterable[O2],
        o3: Iterable[O3],
        o4: Iterable[O4],
        /,
    ) -> PyoIterator[S | O1 | O2 | O3 | O4]: ...
    @overload
    def chain[S, O1, O2, O3, O4, O5](
        self: PyoIterator[S],
        o1: Iterable[O1],
        o2: Iterable[O2],
        o3: Iterable[O3],
        o4: Iterable[O4],
        o5: Iterable[O5],
        /,
    ) -> PyoIterator[S | O1 | O2 | O3 | O4 | O5]: ...
    def chain[S, O](self: PyoIterator[S], *others: Iterable[O]) -> PyoIterator[S | O]:
        """Concatenate **self** with one or more `Iterables`, any of which may be infinite.

        In other words, it links **self** and **others** together, in a chain. 🔗

        An infinite `Iterable` will prevent the rest of the arguments from being included.

        This is equivalent to `list.extend()`, except it is fully lazy and works with any `Iterable`.

        Tip:
            You can use `Iter.once()` with `chain()` for lazily prepending values to an already existing `Iterator`.

        Args:
            *others (Iterable[O]): Other iterables to concatenate.

        Returns:
            PyoIterator[S | O]: A new `Iterator` which will first iterate over values from the original `Iterator` and then over values from the **others** `Iterable`s.

        Example:
            ```python
            from pyochain import Seq, Iter, Range

            data = Seq(1, 2)
            # Multiple iterables of different types can be chained together:
            mixed = data.iter().chain((3, 4), [True], "hi").collect(Seq)
            assert mixed == (1, 2, 3, 4, True, "h", "i")
            # You can also chain infinite iterators,
            chained = (
                data
                .iter()
                .chain(Iter.from_count(3))
                .chain(Iter.from_count(2).map(lambda _: "unreachable"))
                .take(5)
                .collect(Seq)
            )
            assert chained == (1, 2, 3, 4, 5)
            ```
        """

    def collect[R: Collection[Any]](self, collector: Callable[[Iterator[T]], R]) -> R:
        """Transforms the `Iterator` into a collection.

        The most basic pattern in which `collect()` is used is to turn one collection into another.

        You take a collection, call `iter()` on it, do a bunch of transformations, and then `collect()` at the end.

        You specify the target `Collection` type by providing a **collector** function or type.

        This can be any `Callable` that takes an `Iterator[T]` and returns a `Collection[T]` of those types.

        This is equivalent to `Pipe::pipe` at runtime, but with a few differences:

            - A narrower constraint (`Collection[Any]`) to specify the intent
            - Better performance (no args/kwargs unpacking).

        If you need to pass additional arguments, you can use [`Pipe::pipe`][Pipe.pipe] instead.

        Args:
            collector (Callable[[Iterator[T]], R]): Function|type that defines the target collection.

        Returns:
            R: A materialized `Collection` containing the collected elements.

        Example:
            ```python
            from pyochain import Iter, Range, Vec, Dict

            data = Range(0, 4)
            assert data.iter().collect(list) == [0, 1, 2, 3]
            assert data.iter().collect(Vec) == Vec(0, 1, 2, 3)
            assert data.iter().map(str).enumerate().collect(dict) == {
                0: "0",
                1: "1",
                2: "2",
                3: "3",
            }
            ```
            Sometimes type checkers can't infer the type of the collector, in which case you can use an explicit type annotation to help them out.

            In the example below, without the annotation in `collect()`,

            BasedPyright infer `data` as `Seq[Result[int, Any] | Result[Any, int]]` because of the conditional expression in the `map()`, which is not very useful.
            ```python
            from pyochain import Range, Seq, Ok, Err, Result

            data = (
                Range(0, 5)
                .iter()
                .map(lambda x: Ok(x) if x % 2 == 0 else Err(x))
                .collect(Seq[Result[int, int]])
            )
            assert data.pipe(repr) == "Seq(Ok(0), Err(1), Ok(2), Err(3), Ok(4))"
            ```
            Strictly speaking, this is equivalent to annotating the variable at the beginning, but some may prefer this style to keep the type information close to the actual collection operation.

            This notably avoid repetition if you collect anything else than the default `Seq` type.
        """

    @overload
    def collect_into[S](self: PyoIterator[S], collection: Vec[S]) -> Vec[S]: ...
    @overload
    def collect_into[S](
        self: PyoIterator[S], collection: PyoMutableSequence[S]
    ) -> PyoMutableSequence[S]: ...
    @overload
    def collect_into[S](self: PyoIterator[S], collection: list[S]) -> list[S]: ...
    def collect_into(self, collection: MutableSequence[T]) -> MutableSequence[T]:
        """Collects all the items from the `Iterator` into a `MutableSequence`.

        The `MutableSequence` is then returned, so the call chain can be continued.

        This is useful when you already have a `MutableSequence` and want to add the `Iterator` items to it.

        This method is a convenience method to call `MutableSequence.extend()`, but instead of being called on a `MutableSequence`, it's called on an `Iterator`.

        Args:
            collection (MutableSequence[T]): A mutable collection to collect items into.

        Returns:
            MutableSequence[T]: The same mutable collection passed as argument, now containing the collected items.

        Example:
            Basic usage:
            ```python
            from pyochain import Seq, Iter, Vec

            a = Seq(2, 3)
            vec = Vec(1)
            b = a.iter().map(lambda x: x * 2).collect_into(vec)
            assert b == Vec(1, 4, 6)
            c = a.iter().map(lambda x: x * 10).collect_into(vec)
            assert c == Vec(1, 4, 6, 20, 30)
            ```
            The returned mutable sequence can be used to continue the call chain:
            ```python
            from pyochain import Seq, Vec

            a = Seq(1, 2, 3)
            vec = Vec()
            assert a.iter().collect_into(vec).len() == vec.len()
            assert a.iter().collect_into(vec).len() == vec.len()
            ```
        """

    @overload
    def combinations(self, r: Literal[2]) -> PyoIterator[tuple[T, T]]: ...
    @overload
    def combinations(self, r: Literal[3]) -> PyoIterator[tuple[T, T, T]]: ...
    @overload
    def combinations(self, r: Literal[4]) -> PyoIterator[tuple[T, T, T, T]]: ...
    @overload
    def combinations(self, r: Literal[5]) -> PyoIterator[tuple[T, T, T, T, T]]: ...
    def combinations(self, r: int) -> PyoIterator[tuple[T, ...]]:
        """Return an `Iterator` of `tuple` with **r** elements of type `T`.

        The output is a subsequence of `product()`, keeping only entries that are subsequences of the `Iterator`.

        The length of the output is given by `math.comb()` which computes the following:

        `n! / r! / (n - r)!` when `0 ≤ r ≤ n` or zero when `r > n`.

        The combination `tuples` are emitted in lexicographic order according to the order of the `Iterator`.

        If the latter is sorted, the output tuples will be produced in sorted order.

        Args:
            r (int): Length of each combination.

        Returns:
            PyoIterator[tuple[T, ...]]: An `Iterator` of combinations.

        Example:
            ```python
            from pyochain import Seq, Iter, Range

            a = Iter("ABCD").combinations(2).collect(Seq)
            assert a == (
                ("A", "B"),
                ("A", "C"),
                ("A", "D"),
                ("B", "C"),
                ("B", "D"),
                ("C", "D"),
            )
            b = Range(0, 4).iter().combinations(3).collect(Seq)
            assert b == (
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
            )

            combined = Seq(1, 2, 3).iter().combinations(2).collect(Seq)
            assert combined == ((1, 2), (1, 3), (2, 3))
            ```
        """

    @overload
    def combinations_with_replacement(
        self, r: Literal[2]
    ) -> PyoIterator[tuple[T, T]]: ...
    @overload
    def combinations_with_replacement(
        self, r: Literal[3]
    ) -> PyoIterator[tuple[T, T, T]]: ...
    @overload
    def combinations_with_replacement(
        self,
        r: Literal[4],
    ) -> PyoIterator[tuple[T, T, T, T]]: ...
    @overload
    def combinations_with_replacement(
        self,
        r: Literal[5],
    ) -> PyoIterator[tuple[T, T, T, T, T]]: ...
    def combinations_with_replacement(self, r: int) -> PyoIterator[tuple[T, ...]]:
        """Return r length subsequences of elements from the `Iterator`, allowing individual elements to be repeated more than once.

        The output is a subsequence of `product()` that keeps only entries that are subsequences (with possible repeated elements) of the iterable.

        The number of subsequence returned is:

        `(n + r - 1)! / r! / (n - 1)!` when `n > 0`.

        The combination tuples are emitted in lexicographic order according to the order of the `Iterator`.

        If the `Iterator` is sorted, the output tuples will be produced in sorted order.

        Elements are treated as unique based on their position, not on their value.

        If the input elements are unique, the generated combinations will also be unique.

        Args:
            r (int): Length of each combination.

        Returns:
            PyoIterator[tuple[T, ...]]: An `Iterator` of combinations with replacement.

        Example:
            ```python
            from pyochain import Range, Seq, Iter

            a = Seq(1, 2, 3).iter().combinations_with_replacement(2).collect(Seq)
            assert a == ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
            b = Iter("ABC").combinations_with_replacement(2).collect(Seq)
            assert b == (
                ("A", "A"),
                ("A", "B"),
                ("A", "C"),
                ("B", "B"),
                ("B", "C"),
                ("C", "C"),
            )
            ```
        """

    def compress(self, *selectors: bool) -> PyoIterator[T]:
        """Filter elements using a boolean selector iterable.

        Stops when either the `Iterator` or selectors iterables have been exhausted

        Args:
            *selectors (bool): Boolean values indicating which elements to keep.

        Returns:
            PyoIterator[T]: An `Iterator` of the items selected by the boolean selectors.

        Example:
            ```python
            from pyochain import Iter, Seq

            data = Seq("ABCDEF")
            selectors = (1, 0, 1, 0, 1, 1)
            expected = ("A", "C", "E", "F")
            a = data.iter().compress(*selectors).collect(Seq)
            assert a == expected
            # Roughly equivalent to:
            b = (
                data
                .iter()
                .zip(selectors)
                .filter_star(lambda _, selector: selector)
                .map_star(lambda x, _: x)
                .collect(Seq)
            )
            assert b == expected
            ```
        """

    def count(self) -> int:
        """Consume the `Iterator` and return the number of elements it contained.

        Returns:
            int: The count of elements.

        Example:
            ```python
            from pyochain import Iter

            data = Iter((1, 2, 3))
            assert data.count() == 3
            # data is now empty
            assert data.count() == 0
            ```
        """

    def cycle(self) -> PyoIterator[T]:
        """Yield elements from the `Iterator` endlessly, saving a copy of each call to `next()`.

        When the iterable is exhausted, return elements from the saved copy.

        Thus, instead of stopping once all the elements have been yielded, the iterator will instead start again, from the beginning.

        After iterating again, it will start at the beginning again. And again. And again. Forever.

        Note that in case the original iterator is empty, the resulting iterator will also be empty.

        You can use [`PyoIterator::take`][take] or [`PyoIterator::slice`][slice] to limit the number of items taken.

        See Also:
            [`PyoIterator::repeat`][repeat] to create an `Iterator` from a single element repeatedly.

        Returns:
            PyoIterator[T]: A new `Iterator` that cycles through the elements indefinitely.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(1, 2).iter().cycle().take(5).collect(Seq) == (1, 2, 1, 2, 1)
            assert Seq("ABC").iter().cycle().take(5).join("") == "ABCAB"
            ```
        """

    def enumerate(self, start: int = 0) -> PyoIterator[tuple[int, T]]:
        """Return a `Iterator` of (index, value) pairs.

        Each value in the `Iterator` is paired with its index, starting from 0.

        Tip:
            `PyoIterator::map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            start (int): The starting index.

        Returns:
            PyoIterator[tuple[int, T]]: An `Iterator` of (index, value) pairs.

        Example:
            ```python
            from pyochain import Seq

            data = Seq("apple", "banana", "cherry")
            output = data.iter().enumerate().collect(Seq)
            assert output == Seq((0, "apple"), (1, "banana"), (2, "cherry"))
            output = (
                data
                .iter()
                .enumerate()
                .map_star(lambda idx, val: (idx, val.upper()))
                .collect(Seq)
            )
            assert output == Seq((0, "APPLE"), (1, "BANANA"), (2, "CHERRY"))
            ```
        """

    def eq(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** and *other* contain the same items in the same order.

        Comparison is performed element by element.

        Two `Iterable`s are equal only if:

        - every compared pair of elements is equal
        - and both iterables are exhausted at the same time

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` when both iterables yield the same sequence of values.

        Example:
            ```python
            from pyochain import Range, Seq

            data = Range(1, 4)
            assert data.iter().eq((1, 2, 3)) and data.iter().eq(data)
            assert not data.iter().eq((1, 2, 4))
            assert not data.iter().eq((1, 2))
            ```
        """

    @overload
    def filter[N](self: PyoIterator[N | None], func: None = None) -> PyoIterator[N]: ...
    @overload
    def filter[R](self, func: Callable[[T], TypeIs[R]]) -> PyoIterator[R]: ...
    @overload
    def filter[R](self, func: Callable[[T], TypeGuard[R]]) -> PyoIterator[R]: ...
    @overload
    def filter(self, func: Callable[[T], object] | None) -> PyoIterator[T]: ...
    def filter[R, N](
        self, func: FilterFn[T, R] = None
    ) -> PyoIterator[T] | PyoIterator[R] | PyoIterator[N]:
        """Creates an `Iterator` with an optional closure to determine if an element should be yielded.

        Given an element the closure must return `True` or `False`.

        The returned `Iterator` will yield only the elements for which the closure returns `True`.

        If no closure is provided, the elements are directly evaluated on their truthiness.

        This means that empty collections, `0`, `False`, and `None` will be filtered out.

        The closure can return a `TypeIs` or `TypeGuard` to narrow the type of the returned `Iterator`.

        This won't have any runtime effect, but allows for better type inference.

        Note:
            `Iter.filter(f).next()` is equivalent to `Iter.find(f)`.

        Args:
            func (FilterFn[T, R]): Function to evaluate each item.

        Returns:
            PyoIterator[T] | PyoIterator[R] | PyoIterator[N]: An `Iterator` of the items that satisfy the predicate.

        Example:
            ```python
            from pyochain import Iter, Seq, Some

            data = (1, 2, 3)
            assert Iter(data).filter(lambda x: x > 1).collect(Seq) == Seq(2, 3)
            # See the equivalence of next and find:
            assert Iter(data).filter(lambda x: x > 1).next() == Some(2)
            assert Iter(data).find(lambda x: x > 1) == Some(2)
            # Using TypeIs to narrow type:
            from typing import TypeIs

            def _is_str(x: object) -> TypeIs[str]:
                return isinstance(x, str)

            mixed_data = (1, "two", 3.0, "four")
            assert Iter(mixed_data).filter(_is_str).collect(Seq) == Seq("two", "four")
            maybe_none = (1, None, 3, None)
            assert Iter(maybe_none).filter().collect(Seq) == Seq(1, 3)
            maybe_false = (0, 1, False, 2, "", 3, None)
            assert Iter(maybe_false).filter().collect(Seq) == Seq(1, 2, 3)
            ```
        """

    @overload
    def filter_false[N](
        self: PyoIterator[N | None], func: None = None
    ) -> PyoIterator[None]: ...
    @overload
    def filter_false[U](self, func: Callable[[T], TypeIs[U]]) -> PyoIterator[U]: ...
    @overload
    def filter_false[U](self, func: Callable[[T], TypeGuard[U]]) -> PyoIterator[U]: ...
    @overload
    def filter_false(self, func: Callable[[T], object]) -> PyoIterator[T]: ...
    def filter_false[U](
        self, func: FilterFn[T, U] = None
    ) -> PyoIterator[T] | PyoIterator[U]:
        """Return elements for which **func** predicate is `False`.

        If no closure is provided, returns the elements who return `False` when calling `__bool__` on them.

        Args:
            func (FilterFn[T, U]): Function to evaluate each item.

        Returns:
            PyoIterator[T] | PyoIterator[U]: An `Iterator` of the items that do not satisfy the predicate.

        Example:
            ```python
            from pyochain import Seq, Range

            a = Range(0, 5).iter().filter_false(lambda x: x > 1).collect(Seq)
            assert a == (0, 1)
            b = Seq(1, 4, 6, 3, 8).iter().filter_false(lambda x: x < 5).collect(Seq)
            assert b == Seq(6, 8)
            # Count number of none values
            assert Seq(1, None, 2, None, 3).iter().filter_false().count() == 2
            ```
        """

    def filter_map[R](self, func: Callable[[T], Option[R]]) -> PyoIterator[R]:
        """Creates an iterator that both filters and maps.

        The returned iterator yields only the values for which the supplied closure returns Some(value).

        `filter_map` can be used to make chains of `filter` and map more concise.

        The example below shows how a `map().filter().map()` can be shortened to a single call to `filter_map`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each item.

        Returns:
            PyoIterator[R]: An iterable of the results where func returned `Some`.

        See Also:
            [`PyoIterator::filter`][filter] with no closure provided if you want to filter out Python native `None` values.

        Example:
            ```python
            from pyochain import Result, Ok, Err, Seq

            def _parse(s: str) -> Result[int, str]:
                try:
                    return Ok(int(s))
                except ValueError:
                    return Err(f"Invalid integer, got {s!r}")

            data = Seq("1", "two", "NaN", "four", "5")
            parsed = data.iter().filter_map(lambda s: _parse(s).ok()).collect(Seq)
            assert parsed == Seq(1, 5)
            # Equivalent to:
            parsed = (
                data
                .iter()
                .map(lambda s: _parse(s).ok())
                .filter(lambda s: s.is_some())
                .map(lambda s: s.unwrap())
                .collect(Seq)
            )
            assert parsed == Seq(1, 5)
            ```
        """

    @overload
    def filter_map_star[R](
        self: PyoIterator[tuple[Any]],
        func: Callable[[Any], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, R](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[[T1, T2], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, R](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, R](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], Option[R]],
    ) -> PyoIterator[R]: ...
    @overload
    def filter_map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], Option[R]],
    ) -> PyoIterator[R]: ...
    def filter_map_star[U: Iterable[Any], R](
        self: PyoIterator[U], func: Callable[..., Option[R]]
    ) -> PyoIterator[R]:
        """Creates an iterator that both filters and maps, where each element is an iterable.

        Unlike `.filter_map()`, which passes each element as a single argument, `.filter_map_star()` unpacks each element into positional arguments for the function.

        In short, for each `element` in the sequence, it computes `func(*element)`.

        This is useful after using methods like `zip`, `product`, or `enumerate` that yield tuples.

        Args:
            func (Callable[..., Option[R]]): Function to apply to unpacked elements.

        Returns:
            PyoIterator[R]: An iterable of the results where func returned `Some`.

        Example:
            ```python
            from pyochain import Result, Ok, Err, Seq

            data = Seq(("1", "10"), ("two", "20"), ("3", "thirty"))

            def _parse_pair(s1: str, s2: str) -> Result[tuple[int, int], str]:
                try:
                    return Ok((int(s1), int(s2)))
                except ValueError:
                    return Err(f"Invalid integer pair: {s1!r}, {s2!r}")

            parsed = (
                data
                .iter()
                .filter_map_star(lambda s1, s2: _parse_pair(s1, s2).ok())
                .collect(list)
            )
            assert parsed == [(1, 10)]
            ```
        """

    @overload
    def filter_star[T1, R](
        self: PyoIterator[tuple[T1]], func: Callable[[T1], TypeIs[R]]
    ) -> PyoIterator[tuple[R]]: ...
    @overload
    def filter_star[T1, R](
        self: PyoIterator[tuple[T1]], func: Callable[[T1], TypeGuard[R]]
    ) -> PyoIterator[tuple[R]]: ...
    @overload
    def filter_star[T1](
        self: PyoIterator[tuple[T1]], func: Callable[[T1], object]
    ) -> PyoIterator[tuple[T1]]: ...
    @overload
    def filter_star[T1, T2, R, R2](
        self: PyoIterator[tuple[T1, T2]], func: Callable[[T1, T2], TypeIs[tuple[R, R2]]]
    ) -> PyoIterator[tuple[R, R2]]: ...
    @overload
    def filter_star[T1, T2, R, R2](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[[T1, T2], TypeGuard[tuple[R, R2]]],
    ) -> PyoIterator[tuple[R, R2]]: ...
    @overload
    def filter_star[T1, T2](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[[T1, T2], object],
    ) -> PyoIterator[tuple[T1, T2]]: ...
    @overload
    def filter_star[T1, T2, T3, R, R2, R3](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], TypeIs[tuple[R, R2, R3]]],
    ) -> PyoIterator[tuple[R, R2, R3]]: ...
    @overload
    def filter_star[T1, T2, T3, R, R2, R3](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], TypeGuard[tuple[R, R2, R3]]],
    ) -> PyoIterator[tuple[R, R2, R3]]: ...
    @overload
    def filter_star[T1, T2, T3](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[[T1, T2, T3], object],
    ) -> PyoIterator[tuple[T1, T2, T3]]: ...
    @overload
    def filter_star[T1, T2, T3, T4](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        func: Callable[[T1, T2, T3, T4], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5, T6]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8, T9](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]]: ...
    @overload
    def filter_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], object],
    ) -> PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]]: ...
    def filter_star[U: tuple[Any, ...]](
        self: PyoIterator[U], func: Callable[..., object]
    ) -> PyoIterator[U]:
        """Creates an `Iterator` which uses a closure **func** to determine if an element should be yielded, where each element is an iterable.

        Unlike `.filter()`, which passes each element as a single argument, `.filter_star()` unpacks each element into positional arguments for the **func**.

        In short, for each element in the `Iterator`, it computes `func(*element)``.

        This is useful after using methods like `.zip()`, `.product()`, or `.enumerate()` that yield tuples.

        Args:
            func (Callable[..., object]): Function to evaluate unpacked elements.

        Returns:
            PyoIterator[U]: An `Iterator` of the items that satisfy the predicate.

        Example:
            ```python
            from pyochain import Seq

            data = Seq("apple", "banana", "cherry", "date")
            output = (
                data
                .iter()
                .enumerate()
                .filter_star(lambda index, _: index % 2 == 0)
                .map_star(lambda _, fruit: fruit.title())
                .collect(Seq)
            )
            assert output == ("Apple", "Cherry")
            ```
        """

    def find(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Searches for an element of an iterator that satisfies a `predicate`.

        Takes a closure that returns true or false as `predicate`, and applies it to each element of the iterator.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Option[T]: The first element satisfying the predicate. `Some(value)` if found, `NONE` otherwise.

        Example:
            ```python
            from pyochain import Range, Some

            def gt_five(x: int) -> bool:
                return x > 5

            def gt_nine(x: int) -> bool:
                return x > 9

            data = Range(0, 10)
            assert data.iter().find(predicate=gt_five) == Some(6)
            assert data.iter().find(predicate=gt_nine).unwrap_or("missing") == "missing"
            ```
        """

    def find_map[R](self, func: Callable[[T], Option[R]]) -> Option[R]:
        """Applies function to the elements of the `Iterator` and returns the first Some(R) result.

        `Iter.find_map(f)` is equivalent to `Iter.filter_map(f).next()`.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element, returning an `Option[R]`.

        Returns:
            Option[R]: The first `Some(R)` result from applying `func`, or `NONE` if no such result is found.

        Example:
            ```python
            from pyochain import Seq, Some, NONE, Option

            def _parse(s: str) -> Option[int]:
                try:
                    return Some(int(s))
                except ValueError:
                    return NONE

            assert Seq("lol", "NaN", "2", "5").iter().find_map(_parse) == Some(2)
            ```
        """

    def flat_map[R](self, func: Callable[[T], Iterable[R]]) -> PyoIterator[R]:
        """Creates an iterator that applies a function to each element of the original iterator and flattens the result.

        This is useful when the **func** you want to pass to `.map()` itself returns an iterable, and you want to avoid having nested iterables in the output.

        This is equivalent to calling `.map(func).flatten()`.

        Args:
            func (Callable[[T], Iterable[R]]): Function to apply to each element.

        Returns:
            PyoIterator[R]: An iterable of flattened transformed elements.

        Example:
            ```python
            from pyochain import Range, Seq

            out = Range(1, 4).iter().flat_map(range).collect(Seq)
            assert out == Seq(0, 0, 1, 0, 1, 2)
            ```
        """

    def flatten[U](self: PyoIterator[Iterable[U]]) -> PyoIterator[U]:
        """Creates an `Iterator` that flattens nested structures.

        This is useful when you have an `Iterator` of `Iterable` and you want to remove one level of indirection.

        Returns:
            PyoIterator[U]: An `Iterator` of flattened elements.

        Example:
            Basic usage:
            ```python
            from pyochain import Seq

            data = Seq((1, 2, 3, 4), (5, 6))
            flattened = data.iter().flatten().collect(Seq)
            assert flattened == Seq(1, 2, 3, 4, 5, 6)
            ```
            Mapping and then flattening:
            ```python
            words = Seq("he", "l", "lo!")
            merged = words.iter().flatten().collect(Seq)
            assert merged == Seq("h", "e", "l", "l", "o", "!")
            ```
            Flattening only removes one level of nesting at a time:
            ```python
            d3 = Seq(((1, 2), (3, 4)), ((5, 6), (7, 8)))
            d2 = d3.iter().flatten().collect(Seq)
            assert d2 == Seq((1, 2), (3, 4), (5, 6), (7, 8))
            d1 = d3.iter().flatten().flatten().collect(Seq)
            assert d1 == Seq(1, 2, 3, 4, 5, 6, 7, 8)
            ```
            Here we see that `flatten()` does not perform a “deep” flatten.

            Instead, only **one** level of nesting is removed.

            That is, if you `flatten()` a three-dimensional array, the result will be two-dimensional and not one-dimensional.

            To get a one-dimensional structure, you have to `flatten()` again.

        """

    def fold[B](self, init: B, func: Callable[[B, T], B]) -> B:
        """Fold every element of the `Iterator` into an accumulator by applying an operation, returning the final result.

        Args:
            init (B): Initial value for the accumulator.
            func (Callable[[B, T], B]): Function that takes the accumulator and current element,
                returning the new accumulator value.

        Returns:
            B: The final accumulated value.

        Note:
            This is similar to `reduce()` but with an initial value.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)

            assert data.iter().fold(0, lambda acc, x: acc + x) == 6
            assert data.iter().fold(10, lambda acc, x: acc + x) == 16
            assert Seq("a", "b", "c").iter().fold("", lambda acc, x: acc + x) == "abc"
            ```
        """

    @overload
    def fold_star[**P, B](
        self: PyoIterator[tuple[Any]],
        init: B,
        func: Callable[[Any], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, **P, B](
        self: PyoIterator[tuple[T1, T2]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, **P, B](
        self: PyoIterator[tuple[T1, T2, T3]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, T9, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    @overload
    def fold_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, **P, B](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        init: B,
        func: Callable[Concatenate[B, T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, P], B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B: ...
    def fold_star[U: Iterable[Any], **P, B](
        self: PyoIterator[U],
        init: B,
        func: Callable[..., B],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> B:
        """Fold every element of the `Iterator` into an accumulator by applying an operation, returning the final result.

        Use this when the items of the `Iterator` are themselves iterables (e.g., tuples), and you want to unpack them as arguments to the folding function.

        Args:
            init (B): Initial value for the accumulator.
            func (Callable[..., B]): Function that takes the accumulator and current element, returning the new accumulator value.
            *args (P.args): Additional positional arguments to pass to **func**.
            **kwargs (P.kwargs): Additional keyword arguments to pass to **func**.

        Returns:
            B: The final accumulated value.

        Note:
            This is similar to `PyoIterator::reduce` but with an initial value.

        Example:
            ```python
            from pyochain import Iter, Seq

            data = Seq((1, 2), (3, 4))
            assert data.iter().fold_star(0, lambda acc, x, y: acc + x + y) == 10
            data = Seq(("a", "b"), ("c", "d"))
            assert data.iter().fold_star("", lambda acc, x, y: acc + x + y) == "abcd"
            ```
            You can also pass additional arguments to the folding function:
            ```python
            data = Seq((1, 2), (3, 4))

            def add_with_offset(acc: int, x: int, y: int, offset: int) -> int:
                return acc + x + y + offset

            assert data.iter().fold_star(0, add_with_offset, 10) == 30
            ```
        """

    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the `Iterator` by applying a function to each element in the `Iterable`.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Args:
            func (Callable[Concatenate[T, P], Any]): Function to apply to each element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Example:
            ```python
            from pyochain import Range, Vec

            out = Vec()
            Range(1, 4).iter().for_each(out.append)
            assert out == Vec(1, 2, 3)
            ```
        """

    @overload
    def for_each_star[T1, T2, **P, R](
        self: PyoIterator[tuple[T1, T2]],
        func: Callable[Concatenate[T1, T2, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, **P, R](
        self: PyoIterator[tuple[T1, T2, T3]],
        func: Callable[Concatenate[T1, T2, T3, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4]],
        func: Callable[Concatenate[T1, T2, T3, T4, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, T9, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    @overload
    def for_each_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, **P, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[Concatenate[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None: ...
    def for_each_star[U: tuple[Any, ...], **P, R](
        self: PyoIterator[U],
        func: Callable[..., R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Consume the `Iterator` by applying a function to each unpacked item in the `Iterable` element.

        Is a terminal operation, and is useful for functions that have side effects,
        or when you want to force evaluation of a lazy iterable.

        Each item yielded by the `Iterator` is expected to be an `Iterable` itself (e.g., a tuple or list),
        and its elements are unpacked as arguments to the provided function.

        This is often used after methods like `zip()` or `enumerate()` that yield tuples.

        Args:
            func (Callable[..., R]): Function to apply to each unpacked element.
            *args (P.args): Positional arguments for the function.
            **kwargs (P.kwargs): Keyword arguments for the function.

        Example:
            ```python
            from pyochain import Range, Vec

            vec = Vec()
            Range(1, 5).iter().batched(2).for_each_star(lambda x, y: vec.append(x + y))
            assert vec == Vec(3, 7)
            ```
        """

    def ge(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** is lexicographically greater than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the longer iterable is considered
        greater.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` if **self** is greater than *other*, or equal to it.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)

            assert data.iter().ge((1, 2))
            assert data.iter().ge(data)
            assert not data.iter().ge((1, 2, 4))
            ```
        """

    @overload
    def group_by(self, key: None = None) -> PyoIterator[tuple[T, PyoIterator[T]]]: ...
    @overload
    def group_by[K](
        self, key: Callable[[T], K]
    ) -> PyoIterator[tuple[K, PyoIterator[T]]]: ...
    @overload
    def group_by[K](
        self, key: Callable[[T], K] | None = None
    ) -> PyoIterator[tuple[K, PyoIterator[T]] | tuple[T, PyoIterator[T]]]: ...
    def group_by(
        self,
        key: Callable[[T], Any] | None = None,
    ) -> PyoIterator[tuple[Any | T, PyoIterator[T]]]:
        """Make an `Iterator` that returns consecutive keys and groups from the iterable.

        The values yielded are `(K, PyoIterator[T])` tuples, where the first element is the group key and the second element is an `Iterator` of type `T` over the group values.

        The `Iterator` needs to already be sorted on the same key function.

        This is due to the fact that it generates a new `Group` every time the value of the **key** function changes.

        That behavior differs from SQL's `GROUP BY` which aggregates common elements regardless of their input order.

        Warning:
            You must materialize the second element of the tuple immediately when iterating over groups.

            Because `.group_by()` uses Python's `itertools::groupby` under the hood, each group's iterator shares internal state.

            When you advance to the next group, the previous group's iterator becomes invalid and will yield empty results.

        Args:
            key (Callable[[T], Any] | None): Function computing a key value for each element. If `None`, this defaults to an identity function and returns the element unchanged.

        Returns:
            PyoIterator[tuple[Any | T, PyoIterator[T]]]: An `Iterator` of `(key, value)` tuples.

        Example:
            Simple usage:
            ```python
            from pyochain import Seq

            out = (
                Seq("AAAABBBCCDAABBB")
                .iter()
                .group_by()
                .map_star(lambda k, _: k)
                .collect(tuple)
            )
            assert out == ("A", "B", "C", "D", "A", "B")
            out = (
                Seq("AAAABBBCCD")
                .iter()
                .group_by()
                .map_star(lambda _, g: g.collect(list))
                .collect(tuple)
            )
            assert out == (
                ["A", "A", "A", "A"],
                ["B", "B", "B"],
                ["C", "C"],
                ["D"],
            )
            ```
            `group_by` can let you compute complex operations very easily and efficiently.

            For example, if we want to group even and odd numbers, we can do it like this:
            ```python
            from pyochain import Iter, Dict, Seq
            from operator import itemgetter

            # Example 1: Group even and odd numbers
            res = (
                Iter
                .from_count()  # create an infinite iterator of integers
                .take(8)  # take the first 8
                .map(lambda x: (x % 2 == 0, x))  # map to (is_even, value)
                .sort_by(itemgetter(0))  # sort by is_even
                .iter()  # Since sort collect to a Vec, we need to convert back to Iter
                .group_by(itemgetter(0))  # group by is_even
                # extract values from groups, discarding keys, and materializing them
                .map_star(
                    lambda g, vals: (g, vals.map_star(lambda _, y: y).collect(Seq))
                )
                .collect(Dict)
            )
            assert res == Dict({False: Seq(1, 3, 5, 7), True: Seq(0, 2, 4, 6)})
            ```
            If we have a dataset who's items have a common key and who's already sorted by that key, we can easily perform grouped operations on it, like this:
            ```python
            from pyochain import Seq

            data = Seq(
                {"name": "Alice", "gender": "F"},
                {"name": "Bob", "gender": "M"},
                {"name": "Charlie", "gender": "M"},
                {"name": "Dan", "gender": "M"},
            )
            # group by the gender key, and count the number of people in each group
            output = (
                data
                .iter()
                .group_by(lambda x: x["gender"])
                .map_star(lambda g, vals: (g, vals.count()))
                .collect(Seq)
            )
            assert output == (("F", 1), ("M", 3))
            ```
            However, you must be careful to materialize the group values immediately when iterating over groups, see below how the values of the groups are empty::
            ```python
            from pyochain import Seq

            groups = (
                Seq("a1", "a2", "b1")
                .iter()
                .group_by(lambda x: x[0])
                .collect(Seq)
                .iter()
                .map_star(lambda g, vals: (g, vals.collect(Seq)))
                .collect(Seq)
            )
            assert groups == (("a", Seq()), ("b", Seq()))
            ```
            As such, the correct pattern is the following:
            ```python
            from pyochain import Seq

            groups = (
                Seq("a1", "a2", "b1", "b2")
                .iter()
                .group_by(lambda x: x[0])
                # ✅ Materialize NOW
                .map_star(lambda g, vals: (g, vals.collect(Seq)))
                .collect(Seq)
            )
            assert groups == (("a", Seq("a1", "a2")), ("b", Seq("b1", "b2")))
            ```
        """

    def gt(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** is lexicographically strictly greater than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, the longer iterable is strictly greater than the shorter one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` if **self** compares strictly after *other*.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)
            assert data.iter().gt((1, 2))
            assert not data.iter().gt((1, 2, 9))
            assert not data.iter().gt((1, 2, 3))
            ```
        """

    def intersperse[S](self: PyoIterator[S], element: S) -> PyoIterator[S]:
        """Creates a new `Iterator` which places a copy of separator between adjacent items of the original iterator.

        Args:
            element (S): The element to interpose between items.

        Returns:
            PyoIterator[S]: A new `Iterator` with the element interposed.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)
            # Simple example with numbers
            a = data.iter().intersperse(0).collect(Seq)
            assert a == Seq(1, 0, 2, 0, 3)
            # Useful when chaining with other operations
            assert data.iter().intersperse(5).sum() == 16
            # Inserting separators between groups, then flattening
            a = (
                Seq((1, 2), (3, 4), (5, 6))
                .iter()
                .intersperse([-1])
                .flatten()
                .collect(Seq)
            )
            assert a == Seq(1, 2, -1, 3, 4, -1, 5, 6)
            ```
        """

    def is_sorted[U: SupportsComparison[Any]](
        self: PyoIterator[U], *, reverse: bool = False, strict: bool = False
    ) -> bool:
        """Returns `True` if the items of the `Iterator` are in sorted order.

        The elements of the `Iterator` must support comparison operations.

        The function returns `False` after encountering the first out-of-order item.

        If there are no out-of-order items, the `Iterator` is exhausted.

        Credits to **more-itertools** for the implementation.

        See Also:
            [`PyoIterator::is_sorted_by`][is_sorted_by] if your elements do not support comparison operations directly, or you want to sort based on a specific attribute or transformation.

        Args:
            reverse (bool): Whether to check for descending order.
            strict (bool): Whether to enforce strict sorting (no equal elements).

        Returns:
            bool: `True` if items are sorted according to the criteria, `False` otherwise.

        Example:
            ```python
            from pyochain import Iter

            assert Iter((1, 2, 3, 4, 5)).is_sorted()
            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 2)
            assert data.iter().is_sorted()
            assert not data.iter().is_sorted(strict=True)
            ```
        """

    def is_sorted_by(
        self,
        key: Callable[[T], SupportsComparison[Any]],
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns `True` if the items of the `Iterator` are in sorted order according to the key function.

        The function returns `False` after encountering the first out-of-order item.

        If there are no out-of-order items, the `Iterator` is exhausted.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], SupportsComparison[Any]]): Function to extract a comparison key from each element.
            reverse (bool): Whether to check for descending order.
            strict (bool): Whether to enforce strict sorting (no equal elements).

        Returns:
            bool: `True` if items are sorted according to the criteria, `False` otherwise.

        Example:
            ```python
            from pyochain import Range, Seq

            assert Range(1, 6).iter().map(str).is_sorted_by(int)
            by_int = Seq(1, 5, 3).iter().map(str).is_sorted_by(int, reverse=True)
            assert not by_int
            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            from pyochain import Seq

            data = Seq("1", "2", "2")
            assert data.iter().is_sorted_by(int)
            assert not data.iter().is_sorted_by(int, strict=True)
            ```
        """

    def join(self: PyoIterable[str], sep: str) -> str:
        """Join all elements of the `Iterator` into a single `str`, with a specified separator.

        This is equivalent to the built-in `str.join()` method, but as a method on the `Iterator` itself.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
            ```python
            from pyochain import Iter

            assert Iter(("a", "b", "c")).join("-") == "a-b-c"
            ```
        """

    def last(self) -> T:
        """Consume the `Iterator` and return it's last element.

        Warning:
            This will never return if the `Iterator` is infinite.

        Returns:
            T: The last element of the `Iterator`.

        Example:
            ```python
            from pyochain import Dict, Seq

            data = Dict({"a": 1, "b": 2, "c": 3})
            assert data.iter().last() == "c"
            # If you have a `Sequence`, you can use `PyoSequence::last` instead, which is more efficient.
            assert data.pipe(Seq).last() == "c"
            ```
        """

    def le(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** is lexicographically less than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the shorter iterable is considered smaller.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` if **self** is smaller than *other*, or equal to it.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)
            assert not data.iter().le((1, 2))
            assert data.iter().le((1, 2, 3))
            assert data.iter().le((1, 3))
            ```
        """

    def lt(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** is lexicographically strictly less than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, a shorter iterable is strictly smaller than a longer one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` if **self** compares strictly before *other*.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)
            assert not data.iter().lt((1, 2))
            assert not data.iter().lt((1, 2, 3))
            assert data.iter().lt((1, 3))
            ```
        """

    def map[R](self, func: Callable[[T], R]) -> PyoIterator[R]:
        """Apply a function **func** to each element of the `Iterator`.

        If you are good at thinking in types, you can think of `PyoIterator::map` like this:

        - You have an `Iterator` that gives you elements of some type `A`
        - You want an `Iterator` of some other type `B`
        - Thenyou can use `.map()`, passing a closure **func** that takes an `A` and returns a `B`.

        `PyoIterator::map` is conceptually similar to a for loop.

        However, as `PyoIterator::map` is lazy, it is best used when you are already working with other `PyoIterator` instances.

        If you are doing some sort of looping for a side effect, it is considered more idiomatic to use `PyoIterator.for_each` than `PyoIterator.map().collect(Seq)`.

        Args:
            func (Callable[[T], R]): Function to apply to each element.

        Returns:
            PyoIterator[R]: An iterator of transformed elements.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(1, 2).iter().map(lambda x: x + 1).collect(Seq) == Seq(2, 3)
            # You can use methods on the class rather than on instance for convenience:
            data = Seq("a", "b", "c")

            a = data.iter().map(str.upper).collect(Seq)
            assert a == Seq("A", "B", "C")
            b = data.iter().map(lambda s: s.upper()).collect(Seq)
            assert b == Seq("A", "B", "C")
            ```
        """

    @overload
    def map_juxt[R1, R2](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        /,
    ) -> PyoIterator[tuple[R1, R2]]: ...
    @overload
    def map_juxt[R1, R2, R3](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5, R6]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5, R6, R7]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5, R6, R7, R8]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8, R9](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        func9: Callable[[T], R9],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5, R6, R7, R8, R9]]: ...
    @overload
    def map_juxt[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10](
        self,
        func1: Callable[[T], R1],
        func2: Callable[[T], R2],
        func3: Callable[[T], R3],
        func4: Callable[[T], R4],
        func5: Callable[[T], R5],
        func6: Callable[[T], R6],
        func7: Callable[[T], R7],
        func8: Callable[[T], R8],
        func9: Callable[[T], R9],
        func10: Callable[[T], R10],
        /,
    ) -> PyoIterator[tuple[R1, R2, R3, R4, R5, R6, R7, R8, R9, R10]]: ...
    @overload
    def map_juxt[R](self, *funcs: Callable[[T], R]) -> PyoIterator[tuple[R, ...]]: ...
    def map_juxt(self, *funcs: Callable[[T], Any]) -> PyoIterator[tuple[Any, ...]]:
        """Apply several functions to each item of the `Iterator`.

        Returns a new `Iterator` where each item is a tuple of the results of applying each function to the original item.

        This can be very handy to compute multiple transformations or properties of the same item in a single pass, without needing to iterate multiple times.

        As such, this can be considered as an alternative to various patterns, such as `PyoIterator::{for_each, fold}` with mutable collections, or `PyoIterator::map` followed by `PyoIterator::zip` to combine the results.

        Args:
            *funcs (Callable[[T], Any]): Functions to apply to each item.

        Returns:
            PyoIterator[tuple[Any, ...]]: An iterable of tuples containing the results of each function.

        Example:
            ```python
            from pyochain import Seq

            def is_even(n: int) -> bool:
                return n % 2 == 0

            def is_positive(n: int) -> bool:
                return n > 0

            out = Seq(1, -2, 3).iter().map_juxt(is_even, is_positive).collect(Seq)
            assert out == Seq((False, True), (True, False), (False, True))
            ```
            If you need to pass additional args and kwargs to the functions, you can use `functools::partial` or create curried functions like this:
            ```python
            from pyochain import Range
            from collections.abc import Callable

            def curried_add(a: int) -> Callable[[int], int]:
                def fn(b: int) -> int:
                    return a + b

                return fn

            out = (
                Range(1, 4)
                .iter()
                .map_juxt(curried_add(10), curried_add(20))
                .collect(Seq)
            )
            assert out == Seq((11, 21), (12, 22), (13, 23))
            ```
            You can then combine this with various other methods to perform complex transformations in a clean and efficient way, without needing to iterate multiple times or create intermediate collections.

            Example with `filter_star`:
            ```python
            from pyochain import Range, Seq

            res = (
                Range(0, 5)
                .iter()
                .map_juxt(lambda x: x * 2, lambda x: x**2)
                .filter_star(lambda double, square: double + square <= 5)
                .collect(Seq)
            )
            assert res == Seq((0, 0), (2, 1))
            ```
        """

    @overload
    def map_star[T1, R](
        self: PyoIterator[tuple[T1]], func: Callable[[T1], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, R](
        self: PyoIterator[tuple[T1, T2]], func: Callable[[T1, T2], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, R](
        self: PyoIterator[tuple[T1, T2, T3]], func: Callable[[T1, T2, T3], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, R](
        self: PyoIterator[tuple[T1, T2, T3, T4]], func: Callable[[T1, T2, T3, T4], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5]],
        func: Callable[[T1, T2, T3, T4, T5], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6]],
        func: Callable[[T1, T2, T3, T4, T5, T6], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, R](
        self: PyoIterator[tuple[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10]],
        func: Callable[[T1, T2, T3, T4, T5, T6, T7, T8, T9, T10], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_star[R](
        self: PyoIterator[tuple[Any, ...]],
        func: Callable[..., R],
    ) -> PyoIterator[R]: ...
    def map_star[U: Iterable[Any], R](
        self: PyoIterator[U], func: Callable[..., R]
    ) -> PyoIterator[R]:
        """Applies a function to each element.where each element is an iterable.

        Unlike `.map()`, which passes each element as a single argument, `.map_star()` unpacks each element into positional arguments for the function.

        In short, for each element in the `Iterator`, it computes `func(*element)`.

        This is strictly equivalent to `itertools::starmap(func, self)`.

        Note:
            Always prefer using `.map_star()` over `.map()` when working with `Iterator` of `tuple` elements.

            Not only it is more readable, but it's also much more performant (up to 30% faster in benchmarks).

        Args:
            func (Callable[..., R]): Function to apply to unpacked elements.

        Returns:
            PyoIterator[R]: An `Iterator` of results from applying the function to unpacked elements.

        Example:
            ```python
            from pyochain import Seq

            def make_sku(color: str, size: str) -> str:
                return f"{color}-{size}"

            data = Seq("blue", "red")
            a = data.iter().product(["S", "M"]).map_star(make_sku).collect(Seq)
            assert a == ("blue-S", "blue-M", "red-S", "red-M")
            # This is equivalent to:
            b = data.iter().product(["S", "M"]).map(lambda x: make_sku(*x)).collect(Seq)
            assert b == ("blue-S", "blue-M", "red-S", "red-M")
            ```
        """

    def map_while[R](self, func: Callable[[T], Option[R]]) -> PyoIterator[R]:
        """Creates an `Iterator` that both yields elements based on a predicate and maps.

        `map_while()` takes a closure as an argument.

        It will call this closure on each element of the `Iterator`, and yield elements while it returns `Some(_)`.

        After `NONE` is returned, `PyoIterator::map_while` stops and the rest of the elements are ignored.

        Args:
            func (Callable[[T], Option[R]]): Function to apply to each element that returns `Option[R]`.

        Returns:
            PyoIterator[R]: An `Iterator` of transformed elements until `NONE` is encountered.

        Example:
            ```python
            from pyochain import Iter, Some, NONE, Seq, Option

            def checked_div(x: int) -> Option[int]:
                return Some(16 // x) if x != 0 else NONE

            data = Seq(-1, 4, 0, 1)
            divided = data.iter().map_while(checked_div).collect(Seq)
            assert divided == Seq(-16, 4)
            data = Seq(0, 1, 2, -3, 4, 5, -6)
            # Convert to positive ints, stop at first negative
            converted = (
                data
                .iter()
                .map_while(lambda x: Some(x) if x >= 0 else NONE)
                .collect(Seq)
            )
            assert converted == Seq(0, 1, 2)
            ```
        """

    @overload
    def map_windows[R](
        self, length: Literal[1], func: Callable[[tuple[T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[2], func: Callable[[tuple[T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[3], func: Callable[[tuple[T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[4], func: Callable[[tuple[T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[5], func: Callable[[tuple[T, T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[6], func: Callable[[tuple[T, T, T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[7], func: Callable[[tuple[T, T, T, T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[8], func: Callable[[tuple[T, T, T, T, T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: Literal[9], func: Callable[[tuple[T, T, T, T, T, T, T, T, T]], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self,
        length: Literal[10],
        func: Callable[[tuple[T, T, T, T, T, T, T, T, T, T]], R],
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows[R](
        self, length: int, func: Callable[[tuple[T, ...]], R]
    ) -> PyoIterator[R]: ...
    def map_windows[R](
        self,
        length: int,
        func: Callable[[tuple[Any, ...]], R],
    ) -> PyoIterator[R]:
        """Calls the given *func* for each contiguous window of size *length* over **self**.

        The windows during mapping overlaps.

        The provided function is called with the entire window as a single tuple argument.

        Args:
            length (int): The length of each window.
            func (Callable[[tuple[Any, ...]], R]): Function to apply to each window.

        Returns:
            PyoIterator[R]: An iterator over the outputs of func.

        See Also:
            [`PyoIterator::map_windows_star`][map_windows_star] for a version that unpacks the window into separate arguments.

        Example:
            ```python
            from pyochain import Seq, Range
            import statistics

            data = Seq(1, 2, 3, 4)
            means = data.iter().map_windows(2, statistics.mean).collect(Seq)
            assert means == Seq(1.5, 2.5, 3.5)

            joined = (
                Seq("abcd")
                .iter()
                .map_windows(3, lambda window: "".join(window).upper())
                .collect(Seq)
            )
            assert joined == Seq("ABC", "BCD")

            sum_windows = Range(0, 5).iter().map_windows(4, sum).collect(Seq)
            assert sum_windows == Seq(6, 10)
            ```
        """

    @overload
    def map_windows_star[R](
        self, length: Literal[1], func: Callable[[T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[2], func: Callable[[T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[3], func: Callable[[T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[4], func: Callable[[T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[5], func: Callable[[T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[6], func: Callable[[T, T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[7], func: Callable[[T, T, T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[8], func: Callable[[T, T, T, T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[9], func: Callable[[T, T, T, T, T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    @overload
    def map_windows_star[R](
        self, length: Literal[10], func: Callable[[T, T, T, T, T, T, T, T, T, T], R]
    ) -> PyoIterator[R]: ...
    def map_windows_star[R](
        self, length: int, func: Callable[..., R]
    ) -> PyoIterator[R]:
        """Calls the given *func* for each contiguous window of size *length* over **self**.

        The windows during mapping overlaps.

        The provided function is called with each element of the window as separate arguments.

        Args:
            length (int): The length of each window.
            func (Callable[..., R]): Function to apply to each window.

        Returns:
            PyoIterator[R]: An iterator over the outputs of func.

        See Also:
            [`PyoIterator::map_windows`][map_windows] for a version that passes the entire window as a single tuple argument.

        Example:
            ```python
            from pyochain import Seq, Iter

            a = Iter("abcd").map_windows_star(2, lambda x, y: f"{x}+{y}").collect(Seq)
            assert a == Seq("a+b", "b+c", "c+d")
            b = (
                Seq(1, 2, 3, 4)
                .iter()
                .map_windows_star(2, lambda x, y: x + y)
                .collect(Seq)
            )
            assert b == Seq(3, 5, 7)
            ```
        """

    @overload
    def map_with[T1, R](
        self, func: Callable[[T, T1], R], iterable: Iterable[T1], /
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, R](
        self,
        func: Callable[[T, T1, T2], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, R](
        self,
        func: Callable[[T, T1, T2, T3], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, R](
        self,
        func: Callable[[T, T1, T2, T3, T4], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, T5, R](
        self,
        func: Callable[[T, T1, T2, T3, T4, T5], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> PyoIterator[R]: ...
    @overload
    def map_with[T1, T2, T3, T4, T5, T6, R](
        self,
        func: Callable[[T, T1, T2, T3, T4, T5, T6], R],
        iterable: Iterable[T1],
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        /,
    ) -> PyoIterator[R]: ...
    def map_with[R](
        self, func: Callable[..., R], *iterables: Iterable[Any]
    ) -> PyoIterator[R]:
        """Applies a function to the elements of this `Iterator` and additional iterables.

        The provided function must take as many arguments as the number of iterables provided (including **self**).

        It is then applied to the items from all iterables in parallel.

        The `Iterator` stops when the shortest iterable is exhausted.

        Args:
            func (Callable[..., R]): Function to apply to the elements of the iterables.
            *iterables (Iterable[Any]): Additional iterables to zip with **self**.

        Returns:
            PyoIterator[R]: An `Iterator` of results from applying the function to the elements of the iterables.

        See Also:
            [`PyoIterator::map_juxt`][map_juxt] to apply multiple functions to the same elements of the `Iterator`.

        Example:
            ```python
            from pyochain import Seq
            from dataclasses import dataclass

            @dataclass
            class Triangle:
                x: int
                y: int
                z: int

            x = Seq(1, 2, 3)
            y = [4, 5, 6]
            z = [7, 8, 9]
            output = x.iter().map_with(Triangle, y, z).collect(Seq)
            assert output == Seq(
                Triangle(x=1, y=4, z=7),
                Triangle(x=2, y=5, z=8),
                Triangle(x=3, y=6, z=9),
            )
            output_2 = x.iter().map_with(lambda a, b, c: a + b + c, y, z).collect(Seq)
            assert output_2 == Seq(12, 15, 18)
            ```
        """

    def max[U: SupportsRichComparison](self: PyoIterable[U]) -> U:
        """Return the maximum element of the `Iterator`.

        The elements of the `Iterator` must support comparison operations.

        For comparing elements using a custom **key** function, use [`max_by`][max_by] instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(3, 1, 2).iter().max() == 3
            ```
        """

    def max_by[U: SupportsRichComparison](self, key: Callable[[T], U]) -> T:
        """Return the maximum element of the `Iterator` using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
            ```python
            from pyochain import Seq
            from dataclasses import dataclass

            @dataclass
            class Person:
                name: str
                age: int
                is_student: bool

                def get_discount(self) -> float:
                    return 0.1 if self.is_student else 0.0

            alice = Person("Alice", 30, False)
            bob = Person("Bob", 22, True)
            charlie = Person("Charlie", 25, False)
            persons = Seq(alice, bob, charlie)

            assert persons.iter().max_by(lambda p: p.age).name == "Alice"
            assert persons.iter().max_by(lambda p: p.name).name == "Charlie"
            assert persons.iter().max_by(Person.get_discount).name == "Bob"
            ```
        """

    def min[U: SupportsRichComparison](self: PyoIterable[U]) -> U:
        """Return the minimum of the `Iterator`.

        The elements of the `Iterator` must support comparison operations.

        For comparing elements using a custom **key** function, use [`min_by`][min_by] instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(3, 1, 2).iter().min() == 1
            ```
        """

    def min_by[U: SupportsRichComparison](self, key: Callable[[T], U]) -> T:
        """Return the minimum element of the `Iterator` using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
            ```python
            from pyochain import Seq
            from dataclasses import dataclass

            @dataclass
            class Person:
                name: str
                age: int
                is_student: bool

                def get_discount(self) -> float:
                    return 0.1 if self.is_student else 0.0

            alice = Person("Alice", 30, False)
            bob = Person("Bob", 22, True)
            charlie = Person("Charlie", 25, False)
            persons = Seq(alice, bob, charlie)

            assert persons.iter().min_by(lambda p: p.age).name == "Bob"
            assert persons.iter().min_by(lambda p: p.name).name == "Alice"
            assert persons.iter().min_by(Person.get_discount).name == "Alice"
            ```
        """

    def ne(self, other: Iterable[object]) -> bool:
        """Return `True` if **self** and *other* differ in value or length.

        This is the logical opposite of `eq()`.

        The result becomes `True` as soon as:

        - a pair of compared elements is not equal
        - or one iterable ends before the other

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[object]): Another `Iterable` to compare against.

        Returns:
            bool: `True` when the two iterables are not equal.

        Example:
            ```python
            from pyochain import Range

            data = Range(1, 4)

            assert not data.iter().ne((1, 2, 3))
            assert data.iter().ne((1, 2, 4))
            assert data.iter().ne((1, 2))
            ```
        """

    def next(self) -> Option[T]:
        """Return the next element in the `Iterator`.

        The actual `__next__()` method must be conform to the Python `Iterator` Protocol, and is what will be actually called if you iterate over the `PyoIterator` instance.

        `PyoIterator::next` is a convenience method that wraps the result in an `Option` to handle exhaustion gracefully, for custom use cases.

        Returns:
            Option[T]: The next element in the iterator. `Some[T]`, or `NONE` if the iterator is exhausted.

        Example:
            ```python
            from pyochain import Seq, Some, Null

            it = Seq(1, 2, 3).iter()
            assert it.next() == Some(1)
            assert it.next() == Some(2)
            assert it.next() == Some(3)
            # iterator is now exhausted
            assert it.next() is Null()
            ```
        """

    def nth(self, n: int) -> Option[T]:
        """Return the nth item of the `Iterable` at the specified *n*.

        This is similar to `__getitem__` but for lazy `Iterators`.

        If *n* is out of bounds, returns `NONE`.

        Args:
            n (int): The index of the item to retrieve. It must be a non-negative integer.

        Returns:
            Option[T]: `Some(item)` at the specified *n*.

        Example:
            ```python
            from pyochain import Range

            data = Range(0, 10)
            assert data.iter().nth(1).unwrap() == 1
            assert data.iter().nth(10).is_none()
            ```
        """

    def pairwise(self) -> PyoIterator[tuple[T, T]]:
        """Return successive overlapping pairs from the `Iterator`.

        The number of 2-tuples in the resulting `Iterator` will be one fewer than the number of inputs.

        It will be empty if the current `Iterator` has fewer than two values.

        Returns:
            PyoIterator[tuple[T, T]]: An `Iterator` of pairs of consecutive elements.

        Example:
            ```python
            from pyochain import Seq

            assert Seq(1, 2, 3).iter().pairwise().collect(Seq) == ((1, 2), (2, 3))
            assert Seq("ABCDEFG").iter().pairwise().collect(Seq) == (
                ("A", "B"),
                ("B", "C"),
                ("C", "D"),
                ("D", "E"),
                ("E", "F"),
                ("F", "G"),
            )
            ```
        """

    def partition[S](
        self: PyoIterable[S], predicate: Callable[[S], bool]
    ) -> tuple[Vec[S], Vec[S]]:
        """Consumes the `Iterator`, creating two `Vec` from it.

        The predicate passed to `partition()` can return true, or false.

        `partition` returns a pair, all of the elements for which it returned `True`, and all of the elements for which it returned `False`.

        Args:
            predicate (Callable[[S], bool]): Function to determine partition boundaries.

        Returns:
            tuple[Vec[S], Vec[S]]: The resulting pair of collections

        Example:
            ```python
            from pyochain import Vec, Range

            a, b = Range(1, 6).iter().partition(lambda x: x % 2 == 0)
            assert a == Vec(2, 4)
            assert b == Vec(1, 3, 5)
            ```
        """

    def peekable[S](self: PyoIterator[S]) -> Peekable[S]:
        """Creates an iterator which can use the peek and peek_mut methods to look at the next element of the `Iterator` without consuming it.

        See their documentation for more information.

        Note that the underlying `Iterator` is still advanced when peek or peek_mut are called for the first time.

        In order to retrieve the next element, `next` is called on the underlying `Iterator`, hence any side effects (i.e. anything other than fetching the next value) of the `next` method will occur.

        Returns:
            Peekable[S]: A new `Iterator` that allows peeking at the next element.

        Examples:
            Basic usage:
            ```python
            from pyochain import Range, Some

            xs = Range(1, 4)
            iterator = xs.iter().peekable()

            # peek() lets us see into the future
            assert iterator.peek() == Some(1)
            assert iterator.next() == Some(1)
            assert iterator.next() == Some(2)

            # we can peek() multiple times, the iterator won't advance
            assert iterator.peek() == Some(3)
            assert iterator.peek() == Some(3)
            assert iterator.next() == Some(3)

            # after the iterator is finished, so is peek()
            assert iterator.peek().is_none()
            assert iterator.next().is_none()
            ```
        """

    @overload
    def permutations(self, r: Literal[2]) -> PyoIterator[tuple[T, T]]: ...
    @overload
    def permutations(self, r: Literal[3]) -> PyoIterator[tuple[T, T, T]]: ...
    @overload
    def permutations(self, r: Literal[4]) -> PyoIterator[tuple[T, T, T, T]]: ...
    @overload
    def permutations(self, r: Literal[5]) -> PyoIterator[tuple[T, T, T, T, T]]: ...
    def permutations(self, r: int | None = None) -> PyoIterator[tuple[T, ...]]:
        """Return successive *r* length permutations of elements from the `Iterator`.

        The output is a subsequence of `product()` where entries with repeated elements have been filtered out.

        The length of the output is given by `math.perm()` which computes:

        `n! / (n - r)! when 0 ≤ r ≤ n or zero when r > n.`

        The permutation tuples are emitted in lexicographic order according to the order of the current `Iterator`, i.e `Self`.

        If `Self` is sorted, the output tuples will be produced in sorted order.

        Elements are treated as unique based on their position, not on their value.

        If `Self` elements are unique, there will be no repeated values within a permutation.

        Args:
            r (int | None): Length of each permutation. If not specified or `None`, defaults to the length of `Self`, and all possible full-length permutations are generated.

        Returns:
            PyoIterator[tuple[T, ...]]: An `Iterator` of permutations.

        Example:
            ```python
            from pyochain import Seq, Range

            a = Seq(1, 2, 3).iter().permutations(2).collect(Seq)
            assert a == ((1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2))

            b = Range(0, 3).iter().permutations().collect(Seq)
            assert b == (
                (0, 1, 2),
                (0, 2, 1),
                (1, 0, 2),
                (1, 2, 0),
                (2, 0, 1),
                (2, 1, 0),
            )
            ```
        """

    @overload
    def product(self, /) -> PyoIterator[tuple[T]]: ...
    @overload
    def product[T2](self, iter2: Iterable[T2], /) -> PyoIterator[tuple[T, T2]]: ...
    @overload
    def product[T2, T3](
        self, iter2: Iterable[T2], iter3: Iterable[T3], /
    ) -> PyoIterator[tuple[T, T2, T3]]: ...
    @overload
    def product[T2, T3, T4](
        self, iter2: Iterable[T2], iter3: Iterable[T3], iter4: Iterable[T4], /
    ) -> PyoIterator[tuple[T, T2, T3, T4]]: ...
    @overload
    def product[T2, T3, T4, T5](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5]]: ...
    @overload
    def product[T2, T3, T4, T5, T6](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8, T9](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        iter9: Iterable[T9],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8, T9]]: ...
    @overload
    def product[T2, T3, T4, T5, T6, T7, T8, T9, T10](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        iter6: Iterable[T6],
        iter7: Iterable[T7],
        iter8: Iterable[T8],
        iter9: Iterable[T9],
        iter10: Iterable[T10],
        /,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5, T6, T7, T8, T9, T10]]: ...
    @overload
    def product[S](
        self: PyoIterator[S], *iterables: Iterable[S], repeat: int = ...
    ) -> PyoIterator[tuple[S, ...]]: ...
    def product(
        self, *iterables: Iterable[Any], repeat: int = 1
    ) -> PyoIterator[tuple[Any, ...]]:
        """Computes the Cartesian product with other `Iterable`.

        Roughly equivalent to nested for-loops as an `Iterator` method.

        ```python
        from pyochain import Iter

        assert Iter.once("A").iter().product("B").collect(tuple) == tuple(
            (x, y) for x in "A" for y in "B"
        )
        ```

        The nested loops cycle like an odometer with the rightmost element advancing on every iteration.

        This pattern creates a lexicographic ordering so that if the input iterables are sorted, the product tuples are emitted in sorted order.

        To compute the product of the current `Iterator` with itself, specify the number of repetitions with the optional repeat keyword argument.

        ```python
        from pyochain import Seq

        x = Seq(["A"])
        a = x.iter().product(repeat=4).collect(Seq)
        b = x.iter().product(x, x, x).collect(Seq)
        assert a == b
        ```
        Before `product()` runs, it completely consumes the input iterables, keeping pools of values in memory to generate the products.

        Accordingly, it is only useful with finite inputs.

        Args:
            *iterables (Iterable[Any]): Other iterables to compute the Cartesian product with.
            repeat (int): The number of repetitions of the Cartesian product.

        Returns:
            PyoIterator[tuple[Any, ...]]: An iterable of tuples containing elements from the Cartesian product.

        Example:
            ```python
            from pyochain import Seq, Range, Iter

            colors = Seq("blue", "red")
            sizes = Seq("S", "M")
            a = colors.iter().product(sizes).collect(Seq)
            assert a == (("blue", "S"), ("blue", "M"), ("red", "S"), ("red", "M"))
            b = (
                colors
                .iter()
                .product(sizes)
                .map_star(lambda color, size: f"{color}-{size}")
                .collect(Seq)
            )
            assert b == ("blue-S", "blue-M", "red-S", "red-M")
            c = (
                Range(1, 4)
                .iter()
                .product((10, 20))
                .filter_star(lambda a, b: a * b >= 40)
                .collect(Seq)
            )
            assert c == ((2, 20), (3, 20))
            d = (
                Seq(26, 33)
                .iter()
                .product(("Michael", "Sophie"), ["Engineer"])
                .map_star(
                    lambda age, name, profession: f"{name} is {age} and is {profession}"
                )
                .collect(tuple)
            )
            assert d == (
                "Michael is 26 and is Engineer",
                "Sophie is 26 and is Engineer",
                "Michael is 33 and is Engineer",
                "Sophie is 33 and is Engineer",
            )
            e = Seq("blue", "red").iter().product(repeat=2).collect(Seq)
            assert e == (
                ("blue", "blue"),
                ("blue", "red"),
                ("red", "blue"),
                ("red", "red"),
            )
            ```
        """

    def reduce[S](self: PyoIterator[S], func: Callable[[S, S], S]) -> S:
        """Apply a function of two arguments cumulatively to the items of an iterable, from left to right.

        This effectively reduces the `Iterator` to a single value.

        If initial is present, it is placed before the items of the `Iterator` in the calculation.

        It then serves as a default when the `Iterator` is empty.

        Args:
            func (Callable[[S, S], S]): Function to apply cumulatively to the items of the iterable.

        Returns:
            S: Single value resulting from cumulative reduction.

        Example:
            ```python
            from pyochain import Range

            assert Range(1, 4).iter().reduce(lambda a, b: a + b) == 6
            ```
        """

    def scan[U](self, initial: U, func: Callable[[U, T], Option[U]]) -> PyoIterator[U]:
        """Transform elements by sharing state between iterations.

        `scan` takes two arguments:

            - an **initial** value which seeds the internal state
            - a **func** with two arguments

        The first being a reference to the internal state and the second an iterator element.

        The **func** can assign to the internal state to share state between iterations.

        On iteration, the **func** will be applied to each element of the iterator and the return value from the func, an Option, is returned by the next method.

        Thus the **func** can return `Some(value)` to yield value, or `NONE` to end the iteration.

        Args:
            initial (U): Initial state.
            func (Callable[[U, T], Option[U]]): Function that takes the current state and an item, and returns an Option.

        Returns:
            PyoIterator[U]: An iterable of the yielded values.

        Example:
            ```python
            from pyochain import Some, NONE, Range, Seq, Option

            def accumulate_until_limit(state: int, item: int) -> Option[int]:
                new_state = state + item
                match new_state:
                    case _ if new_state <= 10:
                        return Some(new_state)
                    case _:
                        return NONE

            out = Range(1, 6).iter().scan(0, accumulate_until_limit).collect(Seq)
            assert out == Seq(1, 3, 6, 10)
            ```
        """

    def skip(self, n: int) -> PyoIterator[T]:
        """Create an `Iterator` that skips the first n elements.

        skip(**n**) skips elements until n elements are skipped or the end of the `Iterator` is reached (whichever happens first).

        After that, all the remaining elements are yielded.

        In particular, if the original `Iterator` is too short, then the returned `Iterator` is empty.

        If **n** is negative or zero, the original `Iterator` is returned unchanged.

        Args:
            n (int): Number of elements to skip.

        Returns:
            PyoIterator[T]: An `Iterator` of the remaining elements.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)

            assert data.iter().skip(1).collect(Seq) == Seq(2, 3)
            assert data.iter().skip(5).collect(Seq).is_empty()
            assert data.iter().skip(0).collect(Seq) == Seq(1, 2, 3)
            ```
        """

    def skip_while(self, predicate: Callable[[T], object]) -> PyoIterator[T]:
        """Skip elements from the `Iterator` while the *predicate* is `True`.

        Afterwards, returns every element.

        Note this does not produce any output until the predicate first becomes false, so this `Iterator` may have a lengthy start-up time.

        Note:
            This is strictly equivalent to `itertools::dropwhile(predicate, iterable)`.

        Args:
            predicate (Callable[[T], object]): Function to evaluate each item.

        Returns:
            PyoIterator[T]: An `Iterator` of the items after skipping those for which the predicate is true.

        Example:
            ```python
            from pyochain import Seq

            out = Seq(1, 2, 0, -1).iter().skip_while(lambda x: x > 0).collect(Seq)
            assert out == Seq(0, -1)
            ```
        """

    def slice(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int | None = None,
    ) -> PyoIterator[T]:
        """Make an `Iterator` that returns selected elements from the iterable.

        Works like sequence slicing but does not support negative values for *start*, *stop*, or *step*.

        Elements are returned consecutively unless step is set higher than one which results in items being skipped.

        Args:
            start (int | None): Starting index. If zero or `None`, iteration starts at zero. Otherwise, elements from the `Iterator` are skipped until start is reached
            stop (int | None): Ending index. If `None`, iteration continues until the input is exhausted, if at all. Otherwise, it stops at the specified position.
            step (int | None): Step size for the slice. Defaults to one.

        Returns:
            PyoIterator[T]: An `Iterator` of the sliced items.

        Example:
            ```python
            from pyochain import Seq, Range

            txt = Seq("ABCDEFG")

            assert txt.iter().slice(stop=2).join("") == "AB"
            assert txt.iter().slice(2, 4).join("") == "CD"
            assert txt.iter().slice(2, None).join("") == "CDEFG"
            assert txt.iter().slice(0, None, 2).join("") == "ACEG"

            data = Range(1, 6)

            assert data.iter().slice(1, 4).collect(Seq) == (2, 3, 4)
            assert data.iter().slice(step=2).collect(Seq) == (1, 3, 5)
            assert data.iter().slice().collect(Seq) == (1, 2, 3, 4, 5)
            ```
        """

    def sort[U: SupportsRichComparison](
        self: PyoIterator[U], *, reverse: bool = False
    ) -> Vec[U]:
        """Sort the elements of the `Iterator`.

        The elements must support rich comparison operations (i.e., they must implement the necessary comparison dunder methods).

        This is strictly equivalent to `sorted(iterable, reverse=reverse)`.

        Note:
            This method must consume the entire `Iterator` to perform the sort.

            The result is a new `Vec` over the sorted sequence.

        Args:
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[U]: A `Vec` with elements sorted.

        Example:
            ```python
            from pyochain import Vec

            assert Vec(3, 1, 2).iter().sort() == Vec(1, 2, 3)
            ```
        """

    def sort_by[S](
        self: PyoIterator[S],
        key: Callable[[S], SupportsRichComparison],
        *,
        reverse: bool = False,
    ) -> Vec[S]:
        """Sort the elements of the sequence transformed by the key function.

        Note:
            This method must consume the entire `Iterator` to perform the sort.

            The result is a new `Vec` over the sorted sequence.

        Args:
            key (Callable[[S], SupportsRichComparison]): Function to extract a comparison key from each element.
            reverse (bool): Whether to sort in descending order.

        Returns:
            Vec[S]: A `Vec` with elements sorted.

        Example:
            ```python
            from pyochain import Seq, Vec

            str_numbers = Seq("3", "1", "2")
            assert str_numbers.iter().sort_by(int) == Vec("1", "2", "3")
            assert str_numbers.iter().sort_by(int, reverse=True) == Vec("3", "2", "1")
            from dataclasses import dataclass

            @dataclass
            class Person:
                name: str
                age: int

            peoples = Seq(Person("Alice", 30), Person("Bob", 25), Person("Charlie", 35))
            sorted_names = (
                peoples
                .iter()
                .sort_by(lambda x: x.age)
                .iter()
                .map(lambda x: x.name)
                .collect(Seq)
            )
            assert sorted_names == Seq("Bob", "Alice", "Charlie")
            ```
        """

    def step_by(self, step: int) -> PyoIterator[T]:
        """Creates an `Iterator` starting at the same point, but stepping by the given **step** at each iteration.

        Note:
            The first element of the iterator will always be returned, regardless of the **step** given.

        Args:
            step (int): Step size for selecting items.

        Returns:
            PyoIterator[T]: An `Iterator` of every nth item.

        Example:
            ```python
            from pyochain import Seq

            out = Seq(0, 1, 2, 3, 4, 5).iter().step_by(2).collect(Seq)
            assert out == Seq(0, 2, 4)
            ```
        """

    @overload
    def sum(self: PyoIterator[bool], start: int = 0) -> int: ...
    @overload
    def sum(self: PyoIterator[LiteralInteger], start: int = 0) -> int: ...
    @overload
    def sum[T1: SupportsSumWithNoDefaultGiven](
        self: PyoIterator[T1],
    ) -> T1 | Literal[0]: ...
    @overload
    def sum[A1: SupportsAnyAdd, A2: SupportsAnyAdd](
        self: PyoIterator[A1], start: A2
    ) -> A1 | A2: ...
    def sum[T1: SupportsSumWithNoDefaultGiven, A1: SupportsAnyAdd, A2: SupportsAnyAdd](
        self: PyoIterator[bool | LiteralInteger] | PyoIterator[T1] | PyoIterator[A1],
        start: int | T1 | A2 = 0,
    ) -> int | T1 | A1 | A2:
        """Return the sum of the `Iterator`.

        If the `Iterator` is empty (i.e., yields no elements), return the value of `start` (which defaults to `0`).

        Args:
            start (int | T1 | A2): The value to return if the `Iterator` is empty.

        Returns:
            int | T1 | A1 | A2: The sum of all elements.

        Example:
            ```python
            from pyochain import Vec

            data = Vec(1, 2, 3)

            assert data.iter().sum() == 6
            data.clear()
            assert data.iter().sum() == 0
            assert data.iter().sum(10) == 10
            ```
        """

    def tail(self, n: int) -> PyoIterator[T]:
        """Return an `Iterator` of the last **n** elements of the `Iterator`.

        Args:
            n (int): Number of elements to return.

        Returns:
            PyoIterator[T]: An `Iterator` containing the last **n** elements.

        Example:
            ```python
            from pyochain import Range

            assert Range(0, 10).iter().tail(2).collect(tuple) == (8, 9)
            ```
        """

    def take(self, n: int) -> PyoIterator[T]:
        """Creates an iterator that yields the first n elements, or fewer if the underlying iterator ends sooner.

        `Iter.take(n)` yields elements until n elements are yielded or the end of the iterator is reached (whichever happens first).

        The returned iterator is either:

        - A prefix of length n if the original iterator contains at least n elements
        - All of the (fewer than n) elements of the original iterator if it contains fewer than n elements.

        Args:
            n (int): Number of elements to take.

        Returns:
            PyoIterator[T]: An `Iterator` of the first n items.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)

            assert data.iter().take(2).collect(Seq) == Seq(1, 2)
            assert data.iter().take(5).collect(Seq) == Seq(1, 2, 3)
            ```
        """

    def take_while(self, predicate: Callable[[T], object]) -> PyoIterator[T]:
        """Yield elements from the `Iterator` as long as the predicate evaluates to `True`.

        Args:
            predicate (Callable[[T], object]): Function to evaluate each item.

        Returns:
            PyoIterator[T]: An `Iterator` of the items taken while the predicate is true.

        Example:
            ```python
            from pyochain import Seq

            a = Seq(1, 2, 0).iter().take_while(lambda x: x > 0).collect(Seq)
            assert a == (1, 2)

            b = Seq(1, 4, 6, 3, 8).iter().take_while(lambda x: x < 5).collect(Seq)
            assert b == (1, 4)
            ```
        """

    def tee(self, n: int = 2) -> tuple[PyoIterator[T], ...]:
        """Split `Self` into `n` new independants `Iterators`.

        When the input iterable is already a tee iterator object, all members of the return tuple are constructed as if they had been produced by the upstream tee() call.

        This “flattening step” allows nested tee() calls to share the same underlying data chain and to have a single update step rather than a chain of calls.

        tee iterators are not threadsafe.

        A `RuntimeError` may be raised when simultaneously using iterators returned by the same `tee()` call, even if the original `Iterator` is threadsafe.

        This `Iterator` may require significant auxiliary storage (depending on how much temporary data needs to be stored).

        In general, if one `Iterator` uses most or all of the data before another `Iterator` starts, it is faster to use `collect()` instead of `tee()`.

        Args:
            n (int): The number of new `Iterators` to create. Defaults to 2.

        Returns:
            tuple[PyoIterator[T], ...]: A tuple of `n` new `Iterators` that can be used independently.

        Example:
            ```python
            from pyochain import Seq, Some

            data = Seq(1, 2, 3)
            it1, it2 = data.iter().tee()

            assert it1.collect(Seq) == data
            assert it2.collect(Seq) == data
            ```

            The flattening property makes tee iterators efficiently peekable:
            ```python
            from pyochain import Iter
            from pyochain.abc import PyoIterator

            def lookahead[T](tee_iterator: PyoIterator[T]) -> Option[T]:
                '''Return the next value without moving the input forward'''
                [forked_iterator] = tee_iterator.tee(1)
                return forked_iterator.next()

            iterator = Iter("abcdef")
            # Make the input peekable
            [iterator] = iterator.tee(1)
            # Move the iterator forward
            assert iterator.next() == Some("a")
            # Check next value
            assert lookahead(iterator) == Some("b")
            # Continue moving forward
            assert iterator.next() == Some("b")
            ```
        """

    @overload
    def try_collect[U](self: PyoIterator[Option[U]]) -> Option[Vec[U]]: ...
    @overload
    def try_collect[U, E](self: PyoIterator[Result[U, E]]) -> Option[Vec[U]]: ...
    def try_collect[U](
        self: PyoIterator[Option[U]] | PyoIterator[Result[U, Any]],
    ) -> Option[Vec[U]]:
        """Fallibly transforms **self** into a `Vec`, short circuiting if a failure is encountered.

        `try_collect()` is a variation of `collect()` that allows fallible conversions during collection.

        Its main use case is simplifying conversions from iterators yielding `Option[T]` or `Result[T, E]` into `Option[Vec[T]]`.

        Also, if a failure is encountered during `try_collect()`, the `Iterator` is still valid and may continue to be used, in which case it will continue iterating starting after the element that triggered the failure.

        See the last example below for an example of how this works.

        Note:
            This method return `Vec[U]` instead of being customizable, because the underlying data structure must be mutable in order to build up the collection.

        Returns:
            Option[Vec[U]]: `Some[Vec[U]]` if all elements were successfully collected, or `NONE` if a failure was encountered.

        Example:
            ```python
            from pyochain import Range, Some, Ok, Err, NONE, Vec, Option, Seq, Iter

            # Successfully collecting an iterator of Option[int] into Option[Vec[int]]:
            assert Range(1, 4).iter().map(Some).try_collect().unwrap() == Vec(1, 2, 3)
            # Failing to collect in the same way:
            assert Seq(Some(1), Some(2), NONE, Some(3)).iter().try_collect().is_none()

            # A similar example, but with Result:
            Range(1, 4).iter().map(Ok).try_collect().unwrap() == Vec(1, 2, 3)
            assert Seq(Ok(1), Err("error"), Ok(3)).iter().try_collect().is_none()

            def external_fn(x: int) -> Option[int]:
                if x % 2 == 0:
                    return Some(x)
                return NONE

            assert Range(1, 5).iter().map(external_fn).try_collect().is_none()
            # Demonstrating that the iterator remains usable after a failure:
            it = Iter((Some(1), NONE, Some(3), Some(4)))
            assert it.try_collect().is_none()
            assert it.try_collect().unwrap() == Vec(3, 4)
            ```
        """

    def try_find[E](
        self, predicate: Callable[[T], Result[bool, E]]
    ) -> Result[Option[T], E]:
        """Applies a function returning `Result[bool, E]` to find first matching element.

        Short-circuits: stops at the first successful `True` or on the first error.

        Args:
            predicate (Callable[[T], Result[bool, E]]): Function returning a `Result[bool, E]`.

        Returns:
            Result[Option[T], E]: The first matching element, or the first error.

        Example:
            ```python
            from pyochain import Ok, Result, Err, Range, Some

            def is_even(x: int) -> Result[bool, str]:
                return Ok(x % 2 == 0) if x >= 0 else Err("negative number")

            assert Range(1, 6).iter().try_find(is_even).unwrap().unwrap() == 2
            ```
        """

    def try_fold[B, E](
        self, init: B, func: Callable[[B, T], Result[B, E]]
    ) -> Result[B, E]:
        """Folds every element into an accumulator, short-circuiting on error.

        Applies **func** cumulatively to items and the accumulator.

        If **func** returns an error, stops and returns that error.

        Args:
            init (B): Initial accumulator value.
            func (Callable[[B, T], Result[B, E]]): Function that takes the accumulator and element, returns a `Result[B, E]`.

        Returns:
            Result[B, E]: Final accumulator or the first error.

        Example:
            ```python
            from pyochain import Ok, Err, Result, Range, Iter, Seq

            def checked_add(acc: int, x: int) -> Result[int, str]:
                new_val = acc + x
                if new_val > 100:
                    return Err("overflow")
                else:
                    return Ok(new_val)

            assert Range(1, 4).iter().try_fold(0, checked_add).unwrap() == 6
            error = Iter.from_count(50, -10).take(5).try_fold(0, checked_add)
            assert error.unwrap_err() == "overflow"
            assert Seq().iter().try_fold(0, checked_add).unwrap() == 0
            ```
        """

    def try_for_each[E](self, f: Callable[[T], Result[Any, E]]) -> Result[tuple[()], E]:
        """Applies a fallible function to each item in the `Iterator`, stopping at the first error and returning that error.

        This can also be thought of as the fallible form of `.for_each()`.

        Args:
            f (Callable[[T], Result[Any, E]]): A function that takes an item of type `T` and returns a `Result`.

        Returns:
            Result[tuple[()], E]: Returns `Ok(())` if all applications of **f** were successful (i.e., returned `Ok`), or the first error `E` encountered.

        Example:
            ```python
            from pyochain import Iter, Result, Ok, Err

            def validate_positive(n: int) -> Result[tuple[()], str]:
                if n > 0:
                    return Ok("success")
                return Err(f"Value {n} is not positive")

            assert Iter((1, 2, 3, 4, 5)).try_for_each(validate_positive).is_ok()

            # Short-circuit on first error:
            error = Iter((1, 2, -1, 4)).try_for_each(validate_positive).unwrap_err()
            assert error == "Value -1 is not positive"
            ```
        """

    def try_reduce[S, E](
        self: PyoIterator[S], func: Callable[[S, S], Result[S, E]]
    ) -> Result[Option[S], E]:
        """Reduces elements to a single one, short-circuiting on error.

        Uses the first element as the initial accumulator. If **func** returns an error, stops immediately.

        Args:
            func (Callable[[S, S], Result[S, E]]): Function that reduces two items, returns a `Result[S, E]`.

        Returns:
            Result[Option[S], E]: Final accumulated value or the first error. Returns `Ok(NONE)` for empty iterable.

        Example:
            ```python
            from pyochain import Ok, Err, Result, Range, Seq

            def checked_add(x: int, y: int) -> Result[int, str]:
                if x + y > 100:
                    return Err("overflow")
                else:
                    return Ok(x + y)

            assert Range(1, 4).iter().try_reduce(checked_add).is_ok()
            assert Seq(50, 60).iter().try_reduce(checked_add).is_err()
            assert Range(0, 0).iter().try_reduce(checked_add).unwrap().is_none()
            ```
        """

    def unique(self) -> PyoIterator[T]:
        """Return only unique elements of the `Iterator`.

        This has the same effect as collecting the `Iterator` into a `StableSet` (keeps original ordering), but this returns a new `Iterator`.

        This means that this operation stay lazy, and can be more efficient depending on the situation.

        If you just need unique elements in a collection right away, collecting the `Iterator` into a `set`-like collection may have more raw speed.

        Thus

        Returns:
            PyoIterator[T]: An `Iterator` of the unique items.

        Example:
            ```python
            from pyochain import Vec, Set, Seq

            data = Seq(1, 1, 2, 2, 3, 3)

            assert data.iter().unique().collect(Seq) == Seq(1, 2, 3)
            assert data.pipe(Set).iter().sort() == Vec(1, 2, 3)
            ```
        """

    def unique_by(self, key: Callable[[T], Any]) -> PyoIterator[T]:
        """Return only unique elements of the iterable.

        Args:
            key (Callable[[T], Any]): Function to transform items before comparison.

        Returns:
            PyoIterator[T]: An `Iterator` of the unique items.

        Example:
            ```python
            from pyochain import Seq

            data = Seq("cat", "mouse", "dog", "hen")
            assert data.iter().unique_by(key=len).collect(Seq) == Seq("cat", "mouse")
            ```
        """

    def unpack_into[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Unpack the `Iterator` in the provided *func*, and return the result.

        This is similar to `Pipe::pipe`, but instead of passing `PyoIterator[T]`, we pass the elements inside `PyoIterator[T]`.

        This avoids you to do `iterator.pipe(lambda x: (*x))`, improving performance and readability.

        Note:
            This method will consume the `Iterator`.

        Args:
            func (Callable[Concatenate[T, P], R]): Function to call with the unpacked elements of the `Iterator`.
            *args (P.args): Additional positional arguments to pass to *func*
            **kwargs (P.kwargs): Additional keyword arguments to pass to *func*

        Returns:
            R: The result of calling *func* with the unpacked elements of the `Iterator` and any additional arguments.

        Example:
            ```python
            from pyochain import Seq

            data = Seq(1, 2, 3)

            def foo(*a: int, x: str) -> str:
                return x + str(sum(a))

            assert data.iter().unpack_into(foo, x="Result: ") == "Result: 6"
            # The example below will work, but is not type safe, as the unpacked elements are passed as explicit positional arguments.
            assert data.iter().unpack_into(lambda a, b, c: a + b + c) == 6
            ```
        """

    def unzip[U, V](
        self: PyoIterator[tuple[U, V]],
    ) -> tuple[PyoIterator[U], PyoIterator[V]]:
        """Converts an `Iterator` of pairs into a pair of `Iterator`s.

        This function is, in some sense, the opposite of `PyoIterator::zip`.

        Both `Iterator`s share the same underlying source.

        Values consumed by one `Iterator` remain in the shared buffer until the other `Iterator` consumes them too.

        Returns:
            tuple[PyoIterator[U], PyoIterator[V]]: A tuple containing two `Iterator`s, one for each element of the pairs.

        Example:
            ```python
            from pyochain import Seq

            data = Seq("a", "b", "c")
            left, right = data.iter().enumerate().unzip()

            assert left.collect(Seq) == Seq(0, 1, 2)
            assert right.collect(Seq) == Seq("a", "b", "c")
            ```
        """

    def with_position(self) -> PyoIterator[tuple[Position, T]]:
        """Return an `Iterator` over (`Position`, `T`) tuples.

        The `Position` indicates whether the item `T` is the first, middle, last, or only element in the `Iterator`.

        Returns:
            PyoIterator[tuple[Position, T]]: An `Iterator` of (`Position`, item) tuples.

        Example:
            ```python
            from pyochain import Seq

            data = Seq("a", "b", "c", "d", "e")
            a = data.iter().with_position().collect(Seq)
            assert a == Seq(
                ("first", "a"),
                ("middle", "b"),
                ("middle", "c"),
                ("middle", "d"),
                ("last", "e"),
            )

            b = data.iter().take(1).with_position().collect(Seq)
            assert b == Seq([("only", "a")])

            c = data.iter().take(2).with_position().collect(Seq)
            assert c == Seq(("first", "a"), ("last", "b"))
            ```
        """

    @overload
    def zip(self, /, *, strict: bool = False) -> PyoIterator[tuple[T]]: ...
    @overload
    def zip[T2](
        self, iter2: Iterable[T2], /, *, strict: bool = False
    ) -> PyoIterator[tuple[T, T2]]: ...
    @overload
    def zip[T2, T3](
        self, iter2: Iterable[T2], iter3: Iterable[T3], /, *, strict: bool = False
    ) -> PyoIterator[tuple[T, T2, T3]]: ...
    @overload
    def zip[T2, T3, T4](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
        *,
        strict: bool = False,
    ) -> PyoIterator[tuple[T, T2, T3, T4]]: ...
    @overload
    def zip[T2, T3, T4, T5](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
        *,
        strict: bool = False,
    ) -> PyoIterator[tuple[T, T2, T3, T4, T5]]: ...
    @overload
    def zip[S](
        self: PyoIterator[S], /, *others: Iterable[S], strict: bool = False
    ) -> PyoIterator[tuple[S, ...]]: ...
    def zip(
        self, /, *others: Iterable[Any], strict: bool = False
    ) -> PyoIterator[tuple[Any, ...]]:
        """Yields n-length tuples, where n is the number of iterables passed as positional arguments.

        The i-th element in every tuple comes from the i-th iterable argument to `.zip()`.

        This continues until the shortest argument is exhausted.

        Note:
            `Iter::map_star` can then be used for subsequent operations on the index and value, in a destructuring manner.
            This keep the code clean and readable, without index access like `[0]` and `[1]` for inline lambdas.

        Args:
            *others (Iterable[Any]): Other iterables to zip with.
            strict (bool): If `True` and one of the arguments is exhausted before the others, raise a ValueError.

        Returns:
            PyoIterator[tuple[Any, ...]]: An `Iterator` of tuples containing elements from the zipped `PyoIterator` and other iterables.

        Example:
            ```python
            from pyochain import Seq

            a = Seq(1, 2).iter().zip((10, 20)).collect(Seq)
            assert a == Seq((1, 10), (2, 20))

            b = Seq("a", "b").iter().zip((1, 2, 3)).collect(Seq)
            assert b == Seq(("a", 1), ("b", 2))
            ```
        """

    @overload
    def zip_longest[T2](
        self, iter2: Iterable[T2], /
    ) -> PyoIterator[tuple[Option[T], Option[T2]]]: ...
    @overload
    def zip_longest[T2, T3](
        self, iter2: Iterable[T2], iter3: Iterable[T3], /
    ) -> PyoIterator[tuple[Option[T], Option[T2], Option[T3]]]: ...
    @overload
    def zip_longest[T2, T3, T4](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        /,
    ) -> PyoIterator[tuple[Option[T], Option[T2], Option[T3], Option[T4]]]: ...
    @overload
    def zip_longest[T2, T3, T4, T5](
        self,
        iter2: Iterable[T2],
        iter3: Iterable[T3],
        iter4: Iterable[T4],
        iter5: Iterable[T5],
        /,
    ) -> PyoIterator[
        tuple[Option[T], Option[T2], Option[T3], Option[T4], Option[T5]]
    ]: ...
    def zip_longest(self, *others: Iterable[Any]) -> ZippedLongest[T]:
        """Make an `Iterator` that aggregates elements from each of `Self` and input `Iterable`s.

        If the iterables are of uneven length, missing values are filled-in with `Null`.

        Otherwise, wrap elements in `Some` when they are present.

        Iteration continues until the longest iterable is exhausted.
        If one of the iterables is potentially infinite, then the zip_longest() function should be wrapped with something that limits the number of calls.

        For example, `PyoIterator::islice` or `PyoIterator::take_while`.

        Args:
            *others (Iterable[Any]): Other iterables to zip with.

        Returns:
            ZippedLongest[T]: An `Iterator` of tuples containing optional elements from the zipped iterables.

        Example:
            ```python
            from pyochain import Iter, Some, NONE, Vec, Seq

            out = Seq(1, 2).iter().zip_longest([10]).collect(Vec)
            assert out == [(Some(1), Some(10)), (Some(2), NONE)]

            # Can be combined with try collect to filter out the NONE:
            zipped = out.iter().map(lambda x: Iter(x).try_collect()).collect(Vec)
            assert zipped == [Some(Vec(1, 10)), NONE]
            ```
        """
