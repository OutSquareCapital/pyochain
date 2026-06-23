from collections.abc import Callable, Iterable, Iterator
from typing import Any, Concatenate, Protocol, overload, runtime_checkable

from ._types import SupportsComparison
from ._utils import no_doctest
from ._vec import Vec
from .abc import PyoIterator
from .rs import Checkable, Fluent, Option, Result

type SupportsAnyComparison = SupportsComparison[Any]  # pyright: ignore[reportExplicitAny]

@runtime_checkable
class PyoIterable[T](Fluent, Checkable, Iterable[T], Protocol):
    """Base ABC for all pyochain `Iterables`.

    It's the common API surface shared by:

    - eager `Collections`: `Seq`, `Vec`, `Set`, `SetMut`, `Dict`
    - lazy `Iterator`: `Iter`

    It extends the standard `Iterable[T]` protocol, as well as `Fluent` and `Checkable`.

    All concrete subclasses must implement `__iter__()`.

    Note:
        The difference between an `Iterable` and an `Iterator` is often misunderstood, but it's actually quite simple.

        An `Iterable` is any object that can **create** an `Iterator`.

        It's sole responsbility is to provide an `__iter__` method.

        This method must return an object that have a `__next__` method, which is the actual `Iterator`.

        An `Iterator` is an object that can produce elements one at a time, and can be exhausted.

        When you do a `for x in my_iterable`, Python implicitly calls `my_iterable.__iter__(), and then repeatedly calls `next()` on the resulting `Iterator` to get the elements.

        More concretely, a `list`, for example, is an `Iterable`.

        You can't call `next()` on a `list`, because it don't know how to produce elements by itself, it's primary responsibility being to **store** them.

        However, as soon as you call `map(my_list)`, `[x for x in my_list]`, (*my_list), or any other operation that needs to visit elements, an `Iterator` is created (implicitly or explicitly) from the `list`.

        It's also why `abc::Iterator::__iter__` returns `Self` by convention.

    Example:
        Since it's very straightforward to implement, it can very easily be integrated into business logic classes to provide them with a rich set of methods for free.

        ```python
        >>> from pyochain.abc import PyoIterable
        >>> from dataclasses import dataclass
        >>>
        >>> @dataclass(slots=True)
        ... class ClientRegistry(PyoIterable[str]):
        ...     clients: list[str]
        ...
        ...     def __iter__(self):
        ...         return iter(self.clients)
        >>>
        >>> registry = ClientRegistry(["Alice", "Bob", "Charlie"])
        >>> registry.iter().all(lambda name: name.startswith("A"))
        False
        >>> registry.iter().join(", ")
        'Alice, Bob, Charlie'
        >>> registry.iter().map(str.lower).join(", ")
        'alice, bob, charlie'
        >>> registry.ok_or("Registry is empty").map(lambda s: s.iter().join(", "))
        Ok('Alice, Bob, Charlie')

        ```
    """

    def iter[I](self: PyoIterable[I]) -> PyoIterator[I]:
        """Returns a `PyoIterator` object over the `Iterable`.

        By default, this returns an `Iter`, but can be overriden by concrete subclasses.

        This method is the pyochain equivalent of the `__iter__` dunder method.

        Returns:
            PyoIterator[T]: An `Iterator` over the `Iterable`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> seq = Seq((1, 2, 3))
            >>> iterator = seq.iter()
            >>> iterator.collect(Seq)
            Seq(1, 2, 3)
            >>> # iterator is now empty
            >>> iterator.collect(Seq)
            Seq()

            ```
        """

class PyoIteratorRS[T](PyoIterable[T], Iterator[T], Protocol):
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
        >>> from pyochain import Seq
        >>> from pyochain.abc import PyoIterator
        >>>
        >>> class Count(PyoIterator[int]):
        ...     def __init__(self, start: int = 0):
        ...         self.current = start
        ...
        ...     def __iter__(self):
        ...         return self
        ...
        ...     def __next__(self):
        ...         val = self.current
        ...         self.current += 1
        ...         return val
        >>>
        >>> counter = Count(5)
        >>> counter.next()
        Some(5)
        >>> counter.next()
        Some(6)
        >>> counter.iter().take(3).collect(Seq)
        Seq(7, 8, 9)

        ```
    """

    @no_doctest
    @classmethod
    def _from_iterable[I](cls, iterable: Iterable[I]) -> PyoIterator[I]:
        """Internal constructor.

        Since some methods returns a new `PyoIterator`, we use this, with the assumption that the concrete subclass has an `__init__` that can accept an `Iterable[T]`.

        If you want to implement a different constructor, you will need to override this method with one that can construct new instances from an iterable argument.

        Args:
            iterable (Iterable[I]): An `Iterable` to create the new `PyoIterator` from.

        Returns:
            PyoIterator[I]: A new instance of the concrete `PyoIterator` subclass.

        See Also:
            This is how python standard library handle `collections::abc::Set`, see the first point below `Notes on using Set [...]`:

            https://docs.python.org/3/library/collections.abc.html#examples-and-recipes

        """

    def count(self) -> int:
        """Consume the `Iterator` and return the number of elements it contained.

        Returns:
            int: The count of elements.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> data = Iter((1, 2, 3))
            >>> data.count()
            3
            >>> # data is now empty
            >>> data.count()
            0

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
            >>> from pyochain import Dict, Seq
            >>> data = Dict({"a": 1, "b": 2, "c": 3})
            >>> data.iter().last()
            'c'
            >>> # If you have a `Sequence`, you can use `PyoSequence::last` instead, which is more efficient.
            >>> data.pipe(Seq).last()
            'c'

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
            >>> from pyochain import Iter
            >>> Iter((1, True)).all()
            True
            >>> Iter(()).all()
            True
            >>> Iter((1, 0)).all()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>>
            >>> Iter((2, 4, 6)).all(is_even)
            True
            >>> Iter(("a", "", "c")).all()
            False
            >>> Iter((1, None, 3)).all()
            False

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
            >>> from pyochain import Iter, Range
            >>> Iter((0, 1)).any()
            True
            >>> Range(0, 0).iter().any()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>> Iter((1, 3, 4)).any(is_even)
            True

            ```
        """
    def eq(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** and *other* contain the same items in the same order.

        Comparison is performed element by element.

        Two `Iterable`s are equal only if:

        - every compared pair of elements is equal
        - and both iterables are exhausted at the same time

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` when both iterables yield the same sequence of values.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).eq(Seq((1, 2, 3)))
            True
            >>> Iter((1, 2, 3)).eq((1, 2, 4))
            False
            >>> Iter((1, 2, 3)).eq((1, 2))
            False
            >>> Iter((1, 2)).eq((1, 2, 3))
            False

            ```
        """

    def ne(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** and *other* differ in value or length.

        This is the logical opposite of `eq()`.

        The result becomes `True` as soon as:

        - a pair of compared elements is not equal
        - or one iterable ends before the other

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` when the two iterables are not equal.

        Example:
            ```python
            >>> from pyochain import Iter, Seq
            >>> Iter((1, 2, 3)).ne(Seq((1, 2, 3)))
            False
            >>> Iter((1, 2, 3)).ne((1, 2, 4))
            True
            >>> Iter((1, 2, 3)).ne((1, 2))
            True

            ```
        """

    def le(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically less than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the shorter iterable is considered smaller.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** is smaller than *other*, or equal to it.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).le((1, 2, 3))
            True
            >>> Iter((1, 2, 3)).le((1, 2, 3))
            True
            >>> Iter((1, 3)).le((1, 2, 9))
            False

            ```
        """

    def lt(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically strictly less than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, a shorter iterable is strictly smaller than a longer one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** compares strictly before *other*.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2)).lt((1, 2, 3))
            True
            >>> Iter((1, 2, 3)).lt((1, 2, 3))
            False
            >>> Iter((1, 2, 3)).lt((1, 3))
            True

            ```
        """

    def gt(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically strictly greater than *other*.

        The first differing pair of elements decides the result.

        If all compared elements are equal, the longer iterable is strictly greater than the shorter one.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** compares strictly after *other*.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).gt((1, 2))
            True
            >>> Iter((1, 3)).gt((1, 2, 9))
            True
            >>> Iter((1, 2)).gt((1, 2, 3))
            False

            ```
        """

    def ge(self, other: Iterable[T]) -> bool:
        """Return `True` if **self** is lexicographically greater than or equal to *other*.

        Comparison is performed element by element, like Python sequence ordering.

        The first differing pair decides the result.

        If all compared elements are equal and one iterable ends first, the longer iterable is considered
        greater.

        Note:
            This consumes any `Iterator` instances involved in the comparison,
            including **self** and *other* when *other* is itself an `Iterator`.

        Args:
            other (Iterable[T]): Another `Iterable[T]` to compare against.

        Returns:
            bool: `True` if **self** is greater than *other*, or equal to it.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).ge((1, 2))
            True
            >>> Iter((1, 2, 3)).ge((1, 2, 3))
            True
            >>> Iter((1, 2)).ge((1, 2, 3))
            False

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
            >>> from pyochain import Iter, Seq
            >>> Iter("abcdefghabcd").arg_max()
            7
            >>> Iter((0, 1, 2, 3, 3, 2, 1, 0)).arg_max()
            3

            ```
            Identify the best machine learning model:
            ```python
            >>> models = Seq(("svm", "random forest", "knn", "naïve bayes"))
            >>> accuracy = Seq((68, 61, 84, 72))
            >>> # Most accurate model
            >>> models.get(accuracy.iter().arg_max()).unwrap()
            'knn'
            >>> # Best accuracy
            >>> accuracy.iter().max()
            84

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
            >>> from pyochain import Iter, Seq
            >>> Iter(("a", "bbb", "cc")).arg_max_by(len)
            1
            >>> Iter(("Alice", "bob", "charlie")).arg_max_by(str.lower)
            2

            ```
            Identify the best machine learning model:
            ```python
            >>> models = Seq(("svm", "random forest", "knn", "naïve bayes"))
            >>> accuracy = Seq(("68", "61", "84", "72"))
            >>> # Most accurate model
            >>> models.get(accuracy.iter().arg_max_by(int)).unwrap()
            'knn'
            >>> # Best accuracy
            >>> accuracy.iter().max_by(int)
            '84'

            ```
        """

    def arg_min(self) -> int:
        """Index of the first occurrence of a minimum value in the `Iterator`.

        Credits to **more-itertools** for the examples.

        Returns:
            int: The index of the minimum value.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter("efghabcdijkl").arg_min()
            4
            >>> Iter((3, 2, 1, 0, 4, 2, 1, 0)).arg_min()
            3

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
            >>> from pyochain import Iter, Seq
            >>> Iter(("aaa", "b", "cc")).arg_min_by(len)
            1
            >>> Iter(("Alice", "bob", "Charlie")).arg_min_by(str.lower)
            0

            ```
            Identify the best machine learning model:
            ```python
            >>> def cost(x: int) -> float:
            ...     "Days for a wound to heal given a subject's age."
            ...     return x**2 - 20 * x + 150
            >>>
            >>> labels = Seq(("homer", "marge", "bart", "lisa", "maggie"))
            >>> ages = Seq((35, 30, 10, 9, 1))
            >>> # Fastest healing family member
            >>> labels.get(ages.iter().arg_min_by(cost)).unwrap()
            'bart'
            >>> # Age with fastest healing
            >>> ages.iter().min_by(key=cost)
            10

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
            >>> from pyochain import Iter, Range
            >>> Iter("AaaA").all_equal(key=str.casefold)
            True
            >>> Range(0, 9).iter().all_equal(key=lambda x: x < 10)
            True

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
            >>> from pyochain import Iter, Seq, Set
            >>> Iter("ABCB").all_unique()
            False
            >>> Iter("ABCb").all_unique()
            True
            >>> # Alternative way to check uniqueness by comparing lengths:
            >>> collection = Seq((1, 2, 3, 3))
            >>> collection.len() == collection.pipe(Set).len()
            False

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
            >>> from pyochain import Iter
            >>> Iter("ABCb").all_unique()
            True
            >>> Iter("ABCb").all_unique_by(str.lower)
            False

            ```
        """
    def for_each[**P](
        self,
        func: Callable[Concatenate[T, P], Any],  # pyright: ignore[reportExplicitAny]
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
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3)).for_each(lambda x: print(x + 1))
            2
            3
            4

            ```
        """

    def is_sorted[U: SupportsAnyComparison](
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
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3, 4, 5)).is_sorted()
            True

            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((1, 2, 2))
            >>> data.iter().is_sorted()
            True
            >>> data.iter().is_sorted(strict=True)
            False

            ```
        """
    def is_sorted_by(
        self,
        key: Callable[[T], SupportsAnyComparison],
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns `True` if the items of the `Iterator` are in sorted order according to the key function.

        The function returns `False` after encountering the first out-of-order item.

        If there are no out-of-order items, the `Iterator` is exhausted.

        Credits to **more-itertools** for the implementation.

        Args:
            key (Callable[[T], SupportsAnyComparison]): Function to extract a comparison key from each element.
            reverse (bool): Whether to check for descending order.
            strict (bool): Whether to enforce strict sorting (no equal elements).

        Returns:
            bool: `True` if items are sorted according to the criteria, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter(["1", "2", "3", "4", "5"]).is_sorted_by(int)
            True
            >>> Iter(["5", "4", "3", "1", "2"]).is_sorted_by(int, reverse=True)
            False

            ```
            If strict, tests for strict sorting, that is, returns False if equal elements are found:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq(("1", "2", "2"))
            >>> data.iter().is_sorted_by(int)
            True
            >>> data.iter().is_sorted_by(int, strict=True)
            False

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
            >>> from pyochain import Iter
            >>> Iter(((1, 2), (3, 4))).for_each_star(lambda x, y: print(x + y))
            3
            7

            ```
        """
    def try_for_each[E](self, f: Callable[[T], Result[Any, E]]) -> Result[tuple[()], E]:  # pyright: ignore[reportExplicitAny]
        """Applies a fallible function to each item in the `Iterator`, stopping at the first error and returning that error.

        This can also be thought of as the fallible form of `.for_each()`.

        Args:
            f (Callable[[T], Result[Any, E]]): A function that takes an item of type `T` and returns a `Result`.

        Returns:
            Result[tuple[()], E]: Returns `Ok(())` if all applications of **f** were successful (i.e., returned `Ok`), or the first error `E` encountered.

        Example:
            ```python
            >>> from pyochain import Iter, Result, Ok, Err
            >>> def validate_positive(n: int) -> Result[tuple[()], str]:
            ...     if n > 0:
            ...         return Ok("success")
            ...     return Err(f"Value {n} is not positive")
            >>>
            >>> Iter((1, 2, 3, 4, 5)).try_for_each(validate_positive)
            Ok(())
            >>> # Short-circuit on first error:
            >>> Iter((1, 2, -1, 4)).try_for_each(validate_positive)
            Err('Value -1 is not positive')

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
            >>> from pyochain import Ok, Result, Err, Range
            >>>
            >>> def is_even(x: int) -> Result[bool, str]:
            ...     return Ok(x % 2 == 0) if x >= 0 else Err("negative number")
            >>>
            >>> Range(1, 6).iter().try_find(is_even)
            Ok(Some(2))

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
            >>> from pyochain import Iter, Ok, Err, Result
            >>>
            >>> def checked_add(acc: int, x: int) -> Result[int, str]:
            ...     new_val = acc + x
            ...     if new_val > 100:
            ...         return Err("overflow")
            ...     return Ok(new_val)
            >>>
            >>> Iter((1, 2, 3)).try_fold(0, checked_add)
            Ok(6)
            >>> Iter([50, 40, 20]).try_fold(0, checked_add)
            Err('overflow')
            >>> Iter(()).try_fold(0, checked_add)
            Ok(0)

            ```
        """

    def try_reduce[E](
        self, func: Callable[[T, T], Result[T, E]]
    ) -> Result[Option[T], E]:
        """Reduces elements to a single one, short-circuiting on error.

        Uses the first element as the initial accumulator. If **func** returns an error, stops immediately.

        Args:
            func (Callable[[T, T], Result[T, E]]): Function that reduces two items, returns a `Result[T, E]`.

        Returns:
            Result[Option[T], E]: Final accumulated value or the first error. Returns `Ok(NONE)` for empty iterable.

        Example:
            ```python
            >>> from pyochain import Iter, Ok, Err, Result
            >>>
            >>> def checked_add(x: int, y: int) -> Result[int, str]:
            ...     if x + y > 100:
            ...         return Err("overflow")
            ...     return Ok(x + y)
            >>>
            >>> Iter((1, 2, 3)).try_reduce(checked_add)
            Ok(Some(6))
            >>> Iter([50, 60]).try_reduce(checked_add)
            Err('overflow')
            >>> Iter(()).try_reduce(checked_add)
            Ok(NONE)

            ```
        """

    @overload
    def try_collect[U](self: PyoIterator[Option[U]]) -> Option[Vec[U]]: ...
    @overload
    def try_collect[U, E](self: PyoIterator[Result[U, E]]) -> Option[Vec[U]]: ...
    def try_collect[U](
        self: PyoIterator[Option[U]] | PyoIterator[Result[U, Any]],  # pyright: ignore[reportExplicitAny]
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
            >>> from pyochain import Range, Some, Ok, Err, NONE, Vec, Option, Seq, Iter
            >>> # Successfully collecting an iterator of Option[int] into Option[Vec[int]]:
            >>> Range(1, 4).iter().map(Some).try_collect()
            Some(Vec(1, 2, 3))
            >>> # Failing to collect in the same way:
            >>> Seq((Some(1), Some(2), NONE, Some(3))).iter().try_collect()
            NONE
            >>> # A similar example, but with Result:
            >>> Range(1, 4).iter().map(Ok).try_collect()
            Some(Vec(1, 2, 3))
            >>> Seq((Ok(1), Err("error"), Ok(3))).iter().try_collect()
            NONE
            >>> def external_fn(x: int) -> Option[int]:
            ...     if x % 2 == 0:
            ...         return Some(x)
            ...     return NONE
            >>>
            >>> Range(1, 5).iter().map(external_fn).try_collect()
            NONE
            >>> # Demonstrating that the iterator remains usable after a failure:
            >>> it = Iter((Some(1), NONE, Some(3), Some(4)))
            >>> it.try_collect()
            NONE
            >>> it.try_collect()
            Some(Vec(3, 4))

            ```
        """

    @overload
    def fold_star[**P, B](
        self: PyoIterator[tuple[Any]],  # pyright: ignore[reportExplicitAny]
        init: B,
        func: Callable[[Any], B],  # pyright: ignore[reportExplicitAny]
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
            >>> from pyochain import Iter, Seq
            >>>
            >>> data = Seq(((1, 2), (3, 4)))
            >>> data.iter().fold_star(0, lambda acc, x, y: acc + x + y)
            10
            >>> data = Seq((("a", "b"), ("c", "d")))
            >>> data.iter().fold_star("", lambda acc, x, y: acc + x + y)
            'abcd'

            ```
            You can also pass additional arguments to the folding function:
            ```python
            >>> from pyochain import Iter, Seq
            >>>
            >>> data = Seq(((1, 2), (3, 4)))
            >>> def add_with_offset(acc: int, x: int, y: int, offset: int) -> int:
            ...     return acc + x + y + offset
            >>>
            >>> data.iter().fold_star(0, add_with_offset, 10)
            30

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
            >>> from pyochain import Iter, Range
            >>>
            >>> def gt_five(x: int) -> bool:
            ...     return x > 5
            >>>
            >>> def gt_nine(x: int) -> bool:
            ...     return x > 9
            >>> data = Range(0, 10)
            >>> data.iter().find(predicate=gt_five)
            Some(6)
            >>> data.iter().find(predicate=gt_nine).unwrap_or("missing")
            'missing'

            ```
        """

    def partition(self, predicate: Callable[[T], bool]) -> tuple[Vec[T], Vec[T]]:
        """Consumes the `Iterator`, creating two `Vec` from it.

        The predicate passed to `partition()` can return true, or false.

        `partition` returns a pair, all of the elements for which it returned `True`, and all of the elements for which it returned `False`.

        Args:
            predicate (Callable[[T], bool]): Function to determine partition boundaries.

        Returns:
            tuple[Vec[T], Vec[T]]: The resulting pair of collections

        Example:
            ```python
            >>> from pyochain import Iter
            >>> Iter((1, 2, 3, 4, 5)).partition(lambda x: x % 2 == 0)
            (Vec(2, 4), Vec(1, 3, 5))

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
            >>> from pyochain import Seq

            >>> data = Seq((1, 2, 3))
            >>> def foo(*a: int, x: str) -> str:
            ...     return x + str(sum(a))
            >>> data.iter().unpack_into(foo, x="Result: ")
            'Result: 6'
            >>> # The example below will work, but is not type safe, as the unpacked elements are passed as explicit positional arguments.
            >>> data.iter().unpack_into(lambda a, b, c: a + b + c)
            6

            ```
        """
