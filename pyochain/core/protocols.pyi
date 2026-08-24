"""These ABCs are used to define a common interface for pyochain collection types with a flexible constructor.

The "flexible constructor", is, generally speaking, a `__new__(data, *elements)` method, where `data` can be an `Iterable[T]` or a single element `T`, and `*elements` are any number of `T` to include in the collection.

This means that said constructor can handle a lot of different situations.

This flexibility, while great for ergonomics, has unfortunately a cost: runtime checks and branching, which may not be desirable in performance-critical code.

Thus, this protocol provides various alternatives, with constructors for each use case.

That way, user who want to prioritize performance or explicitely named methods can use these alternative constructors,
while still having the flexibility of `__new__` for more generic use cases.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import type_check_only

class FromIter[T](ABC):
    @staticmethod
    @abstractmethod
    def from_iter[I](iterable: Iterable[I], /) -> FromIter[I]:
        """Create the instance from an `Iterable`.

        Tip:
            For converting a literal `tuple[T, ...]` to a `Seq[T]`, [`of`][FromArgs.of] is more efficient since it directly constructs the instance from the `*elements` tuple.

            Keep in mind that it's an exception rather than the rule.

        Args:
            iterable (Iterable[I]): The iterable to create the instance from.

        Returns:
            FromIter[I]: A new instance containing the elements from the provided `iterable`.

        Example:
            ```python
            from pyochain import Vec, Seq

            assert Vec.from_iter([1, 2, 3]) == Vec(1, 2, 3)
            assert Seq.from_iter((1, 2, 3)) == Seq(1, 2, 3)
            assert Vec.from_iter("hello") == Vec("h", "e", "l", "l", "o")
            ```
        """

@type_check_only
class FromArgs[T](FromIter[T], ABC):
    @staticmethod
    @abstractmethod
    def of[E](*elements: E) -> FromArgs[E]:
        """Create the instance from a variable number of single elements `E`.

        Args:
            *elements (E): The elements to create the instance from.

        Returns:
            FromArgs[E]: A new instance containing the provided `elements`.

        Example:
            ```python
            from pyochain import Vec, Seq

            assert Vec.of(1, 2, 3) == Vec(1, 2, 3)
            assert Seq.of(1, 2, 3) == Seq(1, 2, 3)
            assert Seq.of([1, 2], 3) == Seq([1, 2], 3) == Seq.from_iter([[1, 2], 3])
            assert Seq.of("h", "e", "l", "l", "o") == Seq("h", "e", "l", "l", "o")
            ```
        """

@type_check_only
class FromKwargs[K, V](FromIter[tuple[K, V]], ABC):
    @staticmethod
    @abstractmethod
    def of[E](**elements: E) -> FromKwargs[str, E]:
        """Create the instance from a variable number of keyword arguments.

        Args:
            **elements (E): The elements to create the instance from.

        Returns:
            FromKwargs[str, E]: A new instance containing the provided `elements`.

        Example:
            ```python
            from pyochain import Dict

            d = Dict.of(a=1, b=2)
            assert d == Dict(a=1, b=2)
            assert repr(d) == "Dict('a': 1, 'b': 2)"
            assert Dict.of(a=1, b=2, c=3) == Dict({"a": 1, "b": 2, "c": 3})
            ```
        """

@type_check_only
class Wrapper[T](ABC):
    @staticmethod
    @abstractmethod
    def wrap[W](wrapped: Iterable[W], /) -> Wrapper[W]:
        """Create the instance from a reference to an existing data structure corresponding to this pyochain type.

        E.g, `Vec.wrap(list)` or `Dict.wrap(dict)`.

        If you have an `Iterator`, prefer using `from_iter` instead of `Wrapper.wrap(wrapped_type)`, as it's more verbose and won't be really more efficient.

        It guarantees no-copy behavior, regardless of the mutability of the underlying data structure.

        Thus, it is the most efficient way to create a non-empty pyochain wrapper from an existing corresponding data structure.

        Warning:
            No-copy behavior means that mutable collections will be shared between wrapper <-> wrapped.

            Hence, modifying one will affect the other.

        Args:
            wrapped (Iterable[W]): The object to wrap.

        Returns:
            Wrapper[W]: A new instance wrapping the provided `wrapped` object.

        Example:
            ```python
            from pyochain import Vec, Seq, SetMut
            from pyochain.collections import StableSet, Deque
            from collections import deque

            original_list = [1, 2, 3]
            vec = Vec.wrap(original_list)
            assert vec == Vec(1, 2, 3)
            vec[0] = 10
            assert original_list == [10, 2, 3]

            original_tuple = (1, 2, 3)
            assert Seq.wrap(original_tuple) == Seq(1, 2, 3)

            py_dict = {"Alice": 30, "Bob": 25, "Charlie": 35}
            set_obj = StableSet.wrap(py_dict)
            assert set_obj == StableSet("Alice", "Bob", "Charlie")
            py_dict["David"] = 40
            assert set_obj == StableSet("Alice", "Bob", "Charlie", "David")

            original = deque([1, 2, 3])
            deque_obj = Deque.wrap(original)

            assert deque_obj == Deque(1, 2, 3)
            original.append(4)

            assert deque_obj == Deque(1, 2, 3, 4)

            original_set = {1, 2, 3}
            set_obj = SetMut.wrap(original_set)
            assert set_obj == SetMut(1, 2, 3)
            original_set.add(4)
            assert set_obj == SetMut(1, 2, 3, 4)

            original_dict = {"a": 1, "b": 2, "c": 3}
            ref_dict = Dict.wrap(original_dict)

            assert ref_dict == Dict(a=1, b=2, c=3)
            assert ref_dict.insert("a", 100) == Some(1)
            assert original_dict == {"a": 100, "b": 2, "c": 3}
            ```
        """

class ArgsWrapper[T](FromArgs[T], Wrapper[T], ABC): ...
class KwargsWrapper[K, V](FromKwargs[K, V], Wrapper[K], ABC): ...
