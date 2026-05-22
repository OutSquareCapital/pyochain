from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Concatenate

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
from .._types import SupportsRichComparison
from ..rs import Checkable, Pipeable

if TYPE_CHECKING:
    from .._iter import Iter


class PyoIterable[T](Pipeable, Checkable, Iterable[T], ABC):
    """Base ABC for all pyochain `Iterables`.

    It's the common API surface shared by:

    - eager `Collections`: `Seq`, `Vec`, `Set`, `SetMut`, `Dict`
    - lazy `Iterator`: `Iter`

    It extends the standard `Iterable[T]` protocol, as well as `Pipeable` and `Checkable`.

    All concrete subclasses must implement `__iter__()`.

    Since it's very straightforward to implement, it can very easily be integrated into business logic classes to provide them with a rich set of methods for free.

    Example:
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
        >>> registry.all(lambda name: name.startswith("A"))
        False
        >>> registry.join(", ")
        'Alice, Bob, Charlie'
        >>> registry.iter().map(str.lower).join(", ")
        'alice, bob, charlie'
        >>> registry.ok_or("Registry is empty").map(lambda s: s.join(", "))
        Ok('Alice, Bob, Charlie')

        ```
    """

    # pyrefly: ignore [implicit-any-attribute]
    __slots__ = ()  # pyright: ignore[reportUnannotatedClassAttribute]

    def iter(self) -> Iter[T]:
        """Get an `Iter` over the `Iterable`.

        Call this to switch to lazy evaluation.

        Note:
            Calling this method on a class who is itself an `Iterator` has no effect.

        Returns:
            Iter[T]: An `Iterator` over the `Iterable`. The element type is inferred from the actual subclass.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> seq = Seq((1, 2, 3))
            >>> iterator = seq.iter()
            >>> iterator.collect()
            Seq(1, 2, 3)
            >>> # iterator is now empty
            >>> iterator.collect()
            Seq()

            ```
        """
        from .._iter import Iter

        return Iter(iter(self))

    def unpack_into[**P, R](
        self,
        func: Callable[Concatenate[T, P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Unpack the `Iterable` in the provided *func*, and return the result.

        This is similar to `Pipeable::into`, but instead of passing `Self`, we pass the elements inside `Self`.

        This avoids you to do `iterable.into(lambda x: (*x))`, improving performance and readability.

        Note:
            This method, if called on a lazy `Iterator`, will consume it.

            As such, this can be considered as an alternative `Iter::collect` method.

        Args:
            func (Callable[Concatenate[T, P], R]): Function to call with the unpacked elements of the `Iterable`.
            *args (P.args): Additional positional arguments to pass to *func*
            **kwargs (P.kwargs): Additional keyword arguments to pass to *func*

        Returns:
            R: The result of calling *func* with the unpacked elements of the `Iterable` and any additional arguments.

        Example:
            ```python
            >>> from pyochain import Seq

            >>> data = Seq((1, 2, 3))
            >>> def foo(*a: int, x: str) -> str:
            ...     return x + str(sum(a))
            >>> data.unpack_into(foo, x="Result: ")
            'Result: 6'
            >>> # The example below will work, but is not type safe, as the unpacked elements are passed as explicit positional arguments.
            >>> data.unpack_into(lambda a, b, c: a + b + c)
            6

            ```
        """
        return func(*self, *args, **kwargs)

    def join(self: PyoIterable[str], sep: str) -> str:
        """Join all elements of the `Iterable` into a single `str`, with a specified separator.

        Args:
            sep (str): Separator to use between elements.

        Returns:
            str: The joined string.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq(("a", "b", "c")).join("-")
            'a-b-c'

            ```
        """
        return sep.join(iter(self))

    def first(self) -> T:
        """Return the first element of the `Iterable`.

        By default, this method convert the `Iterable` to an `Iterator` and returns the first element by calling `next()` on it.

        On `PyoSequence` and its subclasses (`Seq`, `Range`, etc.), this is overriden to directly use an efficient `__getitem__` access.

        If you already are using an `Iter`, prefer `Iter.next()` instead, which returns an `Option[T]` to handle exhaustion gracefully.

        Returns:
            T: The first element of the `Iterable`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> data = Seq((1, 2))
            >>> data.first()
            1
            >>> iterator = data.iter()
            >>> iterator.first()
            1
            >>> iterator.first()
            2
            >>> # iterator is now empty, using first again would raise an error
            >>> iterator.next()
            NONE

            ```
        """
        return next(iter(self))

    def second(self) -> T:
        """Return the second element of the `Iterable`.

        Similar to `first()`, see its documentation for details.

        Returns:
            T: The second element of the `Iterable`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((9, 8)).second()
            8

            ```
        """
        seq = iter(self)
        _ = next(seq)
        return next(seq)

    def last(self) -> T:
        """Return the last element of the `Iterable`.

        This is similar to `__getitem__` but works on lazy `Iterators`.

        Returns:
            T: The last element of the `Iterable`.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((7, 8, 9)).last()
            9

            ```
        """
        return tls.last(iter(self))

    def length(self) -> int:
        """Return the length of the `Iterable`.

        By default, this method converts the `Iterable` to an `Iterator` and counts the elements by consuming it.

        This is overriden on `PyoCollection` and its subclasses to directly use an efficient `__len__` access.

        Returns:
            int: The count of elements.

        Example:
            ```python
            >>> from pyochain import Seq, Range, Iter
            >>> Seq((1, 2)).length()
            2
            >>> Range(0, 5).length()
            5
            >>> data = Iter((1, 2, 3))
            >>> data.length()
            3
            >>> # data is now empty
            >>> data.length()
            0

            ```
        """
        return tls.length(iter(self))

    def sum[U: int | bool](self: PyoIterable[U]) -> int:
        """Return the sum of the `Iterable`.

        If the `Iterable` is empty, return 0.

        Returns:
            int: The sum of all elements.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((1, 2, 3)).sum()
            6

            ```
        """
        return sum(iter(self))

    def min[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the minimum of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `min_by()` instead.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Returns:
            U: The minimum value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((3, 1, 2)).min()
            1

            ```
        """
        return min(iter(self))

    def min_by[U: SupportsRichComparison[Any]](self, key: Callable[[T], U]) -> T:
        """Return the minimum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the minimum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the minimum key value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            ...     is_student: bool
            ...
            ...     def get_discount(self) -> float:
            ...         return 0.1 if self.is_student else 0.0
            >>>
            >>> alice = Person("Alice", 30, False)
            >>> bob = Person("Bob", 22, True)
            >>> charlie = Person("Charlie", 25, False)
            >>> persons = Seq((alice, bob, charlie))
            >>>
            >>> persons.min_by(lambda p: p.age).name
            'Bob'
            >>> persons.min_by(lambda p: p.name).name
            'Alice'
            >>> persons.min_by(Person.get_discount).name
            'Alice'

            ```
        """
        return min(iter(self), key=key)

    def max[U: SupportsRichComparison[Any]](self: PyoIterable[U]) -> U:
        """Return the maximum element of the `Iterable`.

        The elements of the `Iterable` must support comparison operations.

        For comparing elements using a custom **key** function, use `max_by()` instead.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Returns:
            U: The maximum value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> Seq((3, 1, 2)).max()
            3

            ```
        """
        return max(iter(self))

    def max_by[U: SupportsRichComparison[Any]](self, key: Callable[[T], U]) -> T:
        """Return the maximum element of the `Iterable` using a custom **key** function.

        If multiple elements are tied for the maximum value, the first one encountered is returned.

        Args:
            key (Callable[[T], U]): Function to extract a comparison key from each element.

        Returns:
            T: The element with the maximum key value.

        Example:
            ```python
            >>> from pyochain import Seq
            >>> from dataclasses import dataclass
            >>>
            >>> @dataclass
            ... class Person:
            ...     name: str
            ...     age: int
            ...     is_student: bool
            ...
            ...     def get_discount(self) -> float:
            ...         return 0.1 if self.is_student else 0.0
            >>>
            >>> alice = Person("Alice", 30, False)
            >>> bob = Person("Bob", 22, True)
            >>> charlie = Person("Charlie", 25, False)
            >>> persons = Seq((alice, bob, charlie))
            >>>
            >>> persons.max_by(lambda p: p.age).name
            'Alice'
            >>> persons.max_by(lambda p: p.name).name
            'Charlie'
            >>> persons.max_by(Person.get_discount).name
            'Bob'

            ```
        """
        return max(iter(self), key=key)

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
            >>> from pyochain import Seq
            >>> Seq((1, True)).all()
            True
            >>> Seq(()).all()
            True
            >>> Seq((1, 0)).all()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>> Seq((2, 4, 6)).all(is_even)
            True

            ```
        """
        if predicate is None:
            return all(iter(self))
        return all(predicate(x) for x in iter(self))

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
            >>> from pyochain import Seq, Range
            >>> Seq((0, 1)).any()
            True
            >>> Range(0, 0).any()
            False
            >>> def is_even(x: int) -> bool:
            ...     return x % 2 == 0
            >>> Seq((1, 3, 4)).any(is_even)
            True

            ```
        """
        if predicate is None:
            return any(iter(self))
        return any(predicate(x) for x in iter(self))

    def all_unique[U](self) -> bool:
        """Returns True if all the elements of **self** are unique.

        The function returns as soon as the first non-unique element is encountered.

        Elements are assumed to be hashable.

        If you need to check uniqueness based on a custom key function, use `PyoIterable::all_unique_by` instead.

        Note:
            - On `PyoSequence` and subclasses, this is overriden to directly use an efficient `set` access and length comparison.
            - On `PyoSet`, `PyoMapping` and their subclasses, this directly returns `True`.

        Returns:
            bool: `True` if all elements are unique, `False` otherwise.

        Example:
            ```python
            >>> from pyochain import Iter, Dict
            >>> Iter("ABCB").all_unique()
            False
            >>> Iter("ABCb").all_unique()
            True
            >>> data = Dict.from_ref({1: "a", 2: "a"})
            >>> data.all_unique()
            True
            >>> data.values().all_unique()
            False

            ```
        """
        return tls.all_unique(iter(self))

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
        return tls.all_unique_by(iter(self), key)
