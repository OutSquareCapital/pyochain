from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Collection, Iterable, Iterator
from operator import itemgetter, lt
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    get_origin,
    get_type_hints,
)

import cytoolz as cz

from .._types import SupportsRichComparison
from ._converters import Checkable, Pipeable

if TYPE_CHECKING:
    from .._iter import Iter
    from .._option import Option
    from .._result import Result


class PyoIterable[I: Iterable[Any], T](Pipeable, Checkable, Iterable[T]):
    """Base trait for iterable collection types.

    Foundation for all pyochain collections (`Seq`, `Vec`, `Iter`, `Set`, `SetMut`, `Dict`).

    ##  Type Parameters

    - `I`: Internal storage type (e.g., `list[T]`, `tuple[T, ...]`, `Iterator[T]`, `frozenset[T]`)
    - `T`: Element type


    ##  Required

    - Annotate `_inner` with its concrete type. The factory is auto-extracted.
    - Alternatively, manually define `__init__` accepting `Iterable[T]`.


    ## Features

    - auto-generated optimized `__init__` based on `_inner` annotation
    - Various methods for comparison, aggregation, and element access
    - A generic `__repr__` method
    - __match_args__ for pattern matching on `_inner`
    - All methods from `Pipeable` and `Checkable` mixins traits

    Note:
    - Comparisons consume underlying `Iterator` instances
    - For `Iterator` types, `Checkable` methods always return `Some`/`Ok`

    Args:
        data (Iterable[T]): The data to initialize the Iterable with.

    Raises:
        TypeError: If `_inner` is not annotated with a concrete type, or if instantiated directly.

    Example:
    ```python
    >>> from pyochain import traits
    >>> class MyList[T](traits.PyoIterable[list[T], T]):
    ...     _inner: list[T]  # Required annotation
    >>>
    >>> MyList([1, 2, 3]).sum()
    6
    >>> MyList(["a", "b", "c"]).join("-")
    'a-b-c'
    >>> MyList([1, 2, 3])
    MyList(1, 2, 3)

    ```

    """

    _inner: I
    __slots__ = ("_inner",)
    __match_args__ = ("_inner",)

    def __init_subclass__(cls) -> None:
        """Set up __init__ for the subclass if not manually defined."""
        super().__init_subclass__()

        # Generate optimized __init__ if not manually defined
        if "__init__" not in cls.__dict__ and cls is not PyoIterable:
            inner_annotation: Callable[[Iterable[T]], I] | None = get_type_hints(
                cls, localns={"T": Any}
            ).get("_inner", None)

            if inner_annotation is None:
                msg = (
                    f"{cls.__name__} must annotate _inner with its concrete type. "
                    f"Example: _inner: list[T]"
                )
                raise TypeError(msg)

            origin_inner = get_origin(inner_annotation)
            if origin_inner is not None:
                factory_func = origin_inner
            else:
                msg = f"Cannot determine factory from _inner annotation: {inner_annotation}"
                raise TypeError(msg)

            def __init__(self: PyoIterable[I, T], data: Iterable[T]) -> None:  # noqa: N807
                self._inner = factory_func(data)

            cls.__init__ = __init__

    def __init__(self, data: Iterable[T]) -> None:  # noqa: ARG002
        msg = f"{self.__class__.__name__} must be instantiated via a subclass"
        raise TypeError(msg)

    def __iter__(self) -> Iterator[T]:
        """Get an `Iterator[T]` over the _inner `Iterable`."""
        return iter(self._inner)

    def iter(self) -> Iter[T]:
        """Get an iterator over the `Iterable`.

        Call this to switch to lazy evaluation.

        Calling this method with an inner type who's a lazy `Iterator` instance has no effect.

        Returns:
            Iter[T]: An `Iterator` over the `Iterable`. The element type is inferred from the actual subclass.
        """
        from .._iter import Iter

        return Iter(self._inner)

    def __repr__(self) -> str:
        """Provides a generic representation for Iterable types.

        Should be overriden in subclasses that takes more specific _inner types, like polars.Series for example.
        """
        return f"{self.__class__.__name__}({_get_repr(self._inner).unwrap()})"

    def length(self) -> int:
        """Return the length of the Iterable.

        Like the builtin len but works on lazy sequences.

        Returns:
            int: The count of elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).length()
        2

        ```
        """
        return cz.itertoolz.count(self._inner)

    def eq(self, other: Self) -> bool:
        """Check if two Iterables are equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2,3)))
        True
        >>> pc.Iter((1,2,3)).eq(pc.Seq([1,2]))
        False
        >>> pc.Iter((1,2,3)).eq(pc.Iter((1,2)))
        False
        >>> pc.Seq((1,2,3)).eq(pc.Vec([1,2,3]))
        True

        ```
        """
        return tuple(self._inner) == tuple(other._inner)

    def ne(self, other: Self) -> bool:
        """Check if two Iterables are not equal based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data are not equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2)))
        True
        >>> pc.Iter((1,2,3)).ne(pc.Iter((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) != tuple(other._inner)

    def le(self, other: Self) -> bool:
        """Check if this Iterable is less than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is less than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).le(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).le(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) <= tuple(other._inner)

    def lt(self, other: Self) -> bool:
        """Check if this Iterable is less than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is less than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2)).lt(pc.Seq((1,2,3)))
        True
        >>> pc.Seq((1,2,3)).lt(pc.Seq((1,2)))
        False

        ```
        """
        return tuple(self._inner) < tuple(other._inner)

    def gt(self, other: Self) -> bool:
        """Check if this Iterable is greater than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).gt(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).gt(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) > tuple(other._inner)

    def ge(self, other: Self) -> bool:
        """Check if this Iterable is greater than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of self is greater than or equal to that of other, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq((1,2,3)).ge(pc.Seq((1,2)))
        True
        >>> pc.Seq((1,2)).ge(pc.Seq((1,2,3)))
        False

        ```
        """
        return tuple(self._inner) >= tuple(other._inner)

    def join(self: PyoIterable[I, str], sep: str) -> str:
        """Join all elements of the `Iterable` into a single `string`, with a specified separator.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["a", "b", "c"]).join("-")
        'a-b-c'

        ```
        """
        return sep.join(self._inner)

    def reduce(self, func: Callable[[T, T], T]) -> T:
        """Apply a function of two arguments cumulatively to the items of an iterable, from left to right.

        Args:
            func (Callable[[T, T], T]): Function to apply cumulatively to the items of the iterable.

        Returns:
            T: Single value resulting from cumulative reduction.

        This effectively reduces the iterable to a single value.

        If initial is present, it is placed before the items of the iterable in the calculation.

        It then serves as a default when the iterable is empty.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).reduce(lambda a, b: a + b)
        6

        ```
        """
        return functools.reduce(func, self._inner)

    def fold[B](self, init: B, func: Callable[[B, T], B]) -> B:
        """Fold every element into an accumulator by applying an operation, returning the final result.

        Args:
            init (B): Initial value for the accumulator.
            func (Callable[[B, T], B]): Function that takes the accumulator and current element,
                returning the new accumulator value.

        Returns:
            B: The final accumulated value.

        Note:
            This is similar to `reduce()` but with an initial value, making it equivalent to
            Python's `functools.reduce()` with an initializer.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).fold(0, lambda acc, x: acc + x)
        6
        >>> pc.Seq([1, 2, 3]).fold(10, lambda acc, x: acc + x)
        16
        >>> pc.Seq(['a', 'b', 'c']).fold('', lambda acc, x: acc + x)
        'abc'

        ```
        """
        return functools.reduce(func, self._inner, init)

    def first(self) -> T:
        """Return the first element.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The first element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9]).first()
        9

        ```
        """
        return cz.itertoolz.first(self._inner)

    def second(self) -> T:
        """Return the second element.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The second element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9, 8]).second()
        8

        ```
        """
        return cz.itertoolz.second(self._inner)

    def last(self) -> T:
        """Return the last element.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The last element of the iterable.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([7, 8, 9]).last()
        9

        ```
        """
        return cz.itertoolz.last(self._inner)

    def nth(self, index: int) -> T:
        """Return the nth item at index.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            T: The item at the specified index.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([10, 20]).nth(1)
        20

        ```
        """
        return cz.itertoolz.nth(index, self._inner)

    def argmax[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a maximum value in an iterable.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the maximum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq("abcdefghabcd").argmax()
        7
        >>> pc.Seq([0, 1, 2, 3, 3, 2, 1, 0]).argmax()
        3

        ```
        For example, identify the best machine learning model:
        ```python
        >>> models = pc.Seq(["svm", "random forest", "knn", "naïve bayes"])
        >>> accuracy = pc.Seq([68, 61, 84, 72])
        >>> # Most accurate model
        >>> models.nth(accuracy.argmax())
        'knn'
        >>>
        >>> # Best accuracy
        >>> accuracy.into(max)
        84

        ```
        """
        it = self._inner
        if key is not None:
            it = map(key, it)
        return max(enumerate(it), key=itemgetter(1))[0]

    def argmin[U](self, key: Callable[[T], U] | None = None) -> int:
        """Index of the first occurrence of a minimum value in an iterable.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Optional function to determine the value for comparison.

        Returns:
            int: The index of the minimum value.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq("efghabcdijkl").argmin()
        4
        >>> pc.Seq([3, 2, 1, 0, 4, 2, 1, 0]).argmin()
        3

        ```

        For example, look up a label corresponding to the position of a value that minimizes a cost function:
        ```python
        >>> def cost(x):
        ...     "Days for a wound to heal given a subject's age."
        ...     return x**2 - 20 * x + 150
        >>> labels = pc.Seq(["homer", "marge", "bart", "lisa", "maggie"])
        >>> ages = pc.Seq([35, 30, 10, 9, 1])
        >>> # Fastest healing family member
        >>> labels.nth(ages.argmin(key=cost))
        'bart'
        >>> # Age with fastest healing
        >>> ages.into(min, key=cost)
        10

        ```
        """
        it = self._inner
        if key is not None:
            it = map(key, it)
        return min(enumerate(it), key=itemgetter(1))[0]

    def sum[U: int | bool](self: PyoIterable[I, U]) -> int:
        """Return the sum of the `Iterable`.

        If the `Iterable` is empty, return 0.

        Returns:
            int: The sum of all elements.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2, 3]).sum()
        6

        ```
        """
        return sum(self._inner)

    def min[U: SupportsRichComparison[Any]](self: PyoIterable[I, U]) -> U:
        """Return the minimum of the sequence.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `min_by()` instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).min()
        1

        ```
        """
        return min(self._inner)

    def min_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the minimum element using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> pc.Seq([Foo(2, "a"), Foo(1, "b"), Foo(4, "c")]).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')
        >>> pc.Seq([Foo(2, "a"), Foo(1, "b"), Foo(1, "c")]).min_by(key=lambda f: f.x)
        Foo(x=1, y='b')

        ```
        """
        return min(self._inner, key=key)

    def max[U: SupportsRichComparison[Any]](self: PyoIterable[I, U]) -> U:
        """Return the maximum of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `max_by()` instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([3, 1, 2]).max()
        3

        ```
        """
        return max(self._inner)

    def max_by[U: SupportsRichComparison[Any]](self, *, key: Callable[[T], U]) -> T:
        """Return the maximum element using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
        ```python
        >>> import pyochain as pc
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Foo:
        ...     x: int
        ...     y: str
        >>>
        >>> pc.Seq([Foo(2, "a"), Foo(3, "b"), Foo(4, "c")]).max_by(key=lambda f: f.x)
        Foo(x=4, y='c')
        >>> pc.Seq([Foo(2, "a"), Foo(3, "b"), Foo(3, "c")]).max_by(key=lambda f: f.x)
        Foo(x=3, y='b')

        ```
        """
        return max(self._inner, key=key)

    def all(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if every element of the iterator matches a predicate.

        `Iter.all()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if they all return true, then so does `Iter.all()`.

        If any of them return false, it returns false.

        An empty iterator returns true.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if all elements match the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, True]).all()
        True
        >>> pc.Seq([]).all()
        True
        >>> pc.Seq([1, 0]).all()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([2, 4, 6]).all(is_even)
        True

        ```
        """
        if predicate is None:
            return all(self._inner)
        return all(predicate(x) for x in self._inner)

    def any(self, predicate: Callable[[T], bool] | None = None) -> bool:
        """Tests if any element of the iterator matches a predicate.

        `Iter.any()` takes a closure that returns true or false.

        It applies this closure to each element of the iterator, and if any of them return true, then so does `Iter.any()`.

        If they all return false, it returns false.

        An empty iterator returns false.

        Args:
            predicate (Callable[[T], bool] | None): Optional function to evaluate each item.

        Returns:
            bool: True if any element matches the predicate, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([0, 1]).any()
        True
        >>> pc.Seq(range(0)).any()
        False
        >>> def is_even(x: int) -> bool:
        ...     return x % 2 == 0
        >>> pc.Seq([1, 3, 4]).any(is_even)
        True

        ```
        """
        if predicate is None:
            return any(self._inner)
        return any(predicate(x) for x in self._inner)

    def all_equal[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Return True if all items are equal.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all items are equal, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 1, 1]).all_equal()
        True

        ```
        A function that accepts a single argument and returns a transformed version of each input item can be specified with key:
        ```python
        >>> pc.Seq("AaaA").all_equal(key=str.casefold)
        True
        >>> pc.Seq([1, 2, 3]).all_equal(key=lambda x: x < 10)
        True

        ```
        """
        iterator = itertools.groupby(self._inner, key)
        for _first in iterator:
            for _second in iterator:
                return False
            return True
        return True

    def all_unique[U](self, key: Callable[[T], U] | None = None) -> bool:
        """Returns True if all the elements of iterable are unique.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.

        Returns:
            bool: True if all elements are unique, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq("ABCB").all_unique()
        False

        ```
        If a key function is specified, it will be used to make comparisons.
        ```python
        >>> pc.Seq("ABCb").all_unique()
        True
        >>> pc.Seq("ABCb").all_unique(str.lower)
        False

        ```
        The function returns as soon as the first non-unique element is encountered.

        Iterables with a mix of hashable and unhashable items can be used, but the function will be slower for unhashable items

        """
        seenset: set[T | U] = set()
        seenset_add = seenset.add
        seenlist: list[T | U] = []
        seenlist_add = seenlist.append
        for element in map(key, self._inner) if key else self._inner:
            try:
                if element in seenset:
                    return False
                seenset_add(element)
            except TypeError:
                if element in seenlist:
                    return False
                seenlist_add(element)
        return True

    def is_sorted[U](
        self,
        key: Callable[[T], U] | None = None,
        *,
        reverse: bool = False,
        strict: bool = False,
    ) -> bool:
        """Returns True if the items of iterable are in sorted order.

        Credits to more-itertools for the implementation.

        Args:
            key (Callable[[T], U] | None): Function to transform items before comparison. Defaults to None.
            reverse (bool): Whether to check for descending order. Defaults to False.
            strict (bool): Whether to enforce strict sorting (no equal elements). Defaults to False.

        Returns:
            bool: True if items are sorted according to the criteria, False otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq(["1", "2", "3", "4", "5"]).is_sorted(key=int)
        True
        >>> pc.Seq([5, 4, 3, 1, 2]).is_sorted(reverse=True)
        False

        If strict, tests for strict sorting, that is, returns False if equal elements are found:
        ```python
        >>> pc.Seq([1, 2, 2]).is_sorted()
        True
        >>> pc.Seq([1, 2, 2]).is_sorted(strict=True)
        False

        ```

        The function returns False after encountering the first out-of-order item.

        This means it may produce results that differ from the built-in sorted function for objects with unusual comparison dynamics (like math.nan).

        If there are no out-of-order items, the iterable is exhausted.
        """
        it = self._inner if (key is None) else map(key, self._inner)
        a, b = itertools.tee(it)
        next(b, None)
        if reverse:
            b, a = a, b
        return all(map(lt, a, b)) if strict else not any(map(lt, b, a))

    def find(self, predicate: Callable[[T], bool]) -> Option[T]:
        """Searches for an element of an iterator that satisfies a `predicate`.

        Takes a closure that returns true or false as `predicate`, and applies it to each element of the iterator.

        Args:
            predicate (Callable[[T], bool]): Function to evaluate each item.

        Returns:
            Option[T]: The first element satisfying the predicate. `Some(value)` if found, `NONE` otherwise.

        Example:
        ```python
        >>> import pyochain as pc
        >>> def gt_five(x: int) -> bool:
        ...     return x > 5
        >>>
        >>> def gt_nine(x: int) -> bool:
        ...     return x > 9
        >>>
        >>> pc.Seq(range(10)).find(predicate=gt_five)
        Some(6)
        >>> pc.Seq(range(10)).find(predicate=gt_nine).unwrap_or("missing")
        'missing'

        ```
        """
        from .._option import Option

        return Option(next(filter(predicate, self._inner), None))


def _get_repr(data: Iterable[Any]) -> Result[str, str]:
    from pprint import pformat

    from .._result import Err, Ok

    def _repr_inner(data: Collection[Any]) -> Result[str, str]:
        return Ok(pformat(data, sort_dicts=False)[1:-1])

    match data:
        case Iterator():
            return Ok(data.__repr__())
        case Collection():
            match data:
                case set() | frozenset():
                    return _repr_inner(tuple(data))
                case _:
                    match len(data):
                        case 0:
                            return Ok("")
                        case _:
                            return _repr_inner(data)
        case _:
            return Err(
                f"Cannot provide generic representation for PyoIterable with _inner of type {type(data)}"
            )
