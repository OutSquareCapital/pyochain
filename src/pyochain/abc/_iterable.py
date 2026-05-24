from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Concatenate

from .. import _tools as tls  # pyright: ignore[reportMissingModuleSource]
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
