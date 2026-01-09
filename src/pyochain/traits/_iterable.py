from __future__ import annotations

import itertools
from collections.abc import Callable, Collection, Iterable, Iterator
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

            if (
                hasattr(inner_annotation, "__name__")
                and inner_annotation.__name__ == "I"
            ):
                return

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
        return self._inner.__iter__()

    @classmethod
    def new(cls) -> Self:
        """Create an empty `Iterable`.

        Make sure to specify the type when calling this method, e.g., `Vec[int].new()`.

        Otherwise, `T` will be inferred as `Any`.

        This can be very useful for mutable collections like `Vec` and `Dict`.

        However, this can be handy for immutable collections too, for example for reprensenting failure steps in a pipeline.

        Returns:
            Self: A new empty `Iterable` instance.

        Example:
        ```python
        >>> import pyochain as pc
        >>> data = pc.Vec[int].new()
        >>> data
        Vec()
        >>> # Equivalent to
        >>> data: list[str] = []
        >>> data
        []
        >>> my_dict = pc.Dict[str, int].new()
        >>> my_dict.insert("a", 1)
        NONE
        >>> my_dict
        Dict('a': 1)

        ```
        """
        return cls(())

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
        """Return the length of the `Iterable`.

        Like the builtin `len()` function, but works on lazy `Iterators`.

        Returns:
            int: The count of elements.
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).length()
        2
        >>> pc.Iter(range(5)).length()
        5

        ```
        """
        return cz.itertoolz.count(self._inner)

    def eq(self, other: Self) -> bool:
        """Check if two `Iterable`s are equal based on their data.

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
        """Check if two `Iterable`s are not equal based on their data.

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
        """Check if this `Iterable` is less than or equal to another based on their data.

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
        """Check if this `Iterable` is less than another based on their data.

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
        """Check if this `Iterable` is greater than another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of **self** is greater than that of **other**, False otherwise.

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
        """Check if this `Iterable` is greater than or equal to another based on their data.

        Note:
            This will consume any `Iter` instances involved in the comparison (**self** and/or **other**).

        Args:
            other (Self): Another instance of `Self` to compare against.

        Returns:
            bool: True if the underlying data of **self** is greater than or equal to that of **other**, False otherwise.

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
        """Join all elements of the `Iterable` into a single `str`, with a specified separator.

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

    def first(self) -> T:
        """Return the first element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The first element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9]).first()
        9

        ```
        """
        return cz.itertoolz.first(self._inner)

    def second(self) -> T:
        """Return the second element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The second element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([9, 8]).second()
        8

        ```
        """
        return cz.itertoolz.second(self._inner)

    def last(self) -> T:
        """Return the last element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The last element of the `Iterable`.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([7, 8, 9]).last()
        9

        ```
        """
        return cz.itertoolz.last(self._inner)

    def nth(self, index: int) -> T:
        """Return the nth item of the `Iterable` at the specified **index**.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            T: The item at the specified **index**.

        ```python
        >>> import pyochain as pc
        >>> pc.Seq([10, 20]).nth(1)
        20

        ```
        """
        return cz.itertoolz.nth(index, self._inner)

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
        """Return the minimum of the `Iterable`.

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
        """Return the minimum element of the `Iterable` using a custom **key** function.

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
        """Return the maximum element of the `Iterable`.

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
        """Return the maximum element of the `Iterable` using a custom **key** function.

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
        """Tests if every element of the `Iterable` is truthy.

        `Iter.all()` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterable`, and if they all return true, then so does `Iter.all()`.

        If any of them return false, it returns false.

        An empty `Iterable` returns true.

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
        """Tests if any element of the `Iterable` is truthy.

        `Iter.any()` can optionally take a closure that returns true or false.

        It applies this closure to each element of the `Iterable`, and if any of them return true, then so does `Iter.any()`.
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


class PyoCollection[I: Collection[Any], T](PyoIterable[I, T], Collection[T]):
    """Base trait for `Collection` types.

    Foundation for all pyochain collections (`Seq`, `Vec`, `Set`, `SetMut`, `Dict`).

    ##  Type Parameters

    - `I`: Internal storage type (e.g., `list[T]`, `tuple[T, ...]`, `frozenset[T]`)
    - `T`: Element type

    ## Provides
    - `__len__` and `__contains__` implementations based on `_inner`
    - `repeat(n)` method to repeat the entire collection n times in an `Iter`

    """

    _inner: I

    def __len__(self) -> int:
        return self._inner.__len__()

    def __contains__(self, item: object) -> bool:
        return self._inner.__contains__(item)

    def repeat(self, n: int | None = None) -> Iter[Self]:
        """Repeat the entire `Collection` **n** times (as elements) in an `Iter`.

        If **n** is `None`, repeat indefinitely.

        Warning:
            If **n** is `None`, this will create an infinite `Iterator`.

            Be sure to use `Iter.take()` or `Iter.slice()` to limit the number of items taken.

        See Also:
            `Iter.cycle()` to repeat the *elements* of the `Iter` indefinitely (`Iter[T]`).

        Args:
            n (int | None): Optional number of repetitions.

        Returns:
            Iter[Self]: An `Iter` of repeated `Iter`.

        Example:
        ```python
        >>> import pyochain as pc
        >>> pc.Seq([1, 2]).repeat(3).collect()
        Seq(Seq(1, 2), Seq(1, 2), Seq(1, 2))
        >>> pc.Seq(("a", "b")).repeat(2).collect()
        Seq(Seq('a', 'b'), Seq('a', 'b'))
        >>> pc.Seq([0]).repeat().flatten().take(5).collect()
        Seq(0, 0, 0, 0, 0)

        ```
        """
        from .._iter import Iter

        if n is None:
            return Iter(itertools.repeat(self))
        return Iter(itertools.repeat(self, n))


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
