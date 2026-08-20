from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import type_check_only

from pyochain.abc import PyoIterable

@type_check_only
class FlexibleInit[T](PyoIterable[T], ABC):
    """This ABC is used to define a common interface for pyochain collection types with a flexible constructor.

    The "flexible constructor", is, generally speaking, a `__new__(data, *elements)` method, where `data` can be an `Iterable` or a single element, and `*elements` are additional elements to include in the collection.

    This means that said constructor can handle a lot of different situations.

    This flexibility, while great for ergonomics, has unfortunately a cost: runtime checks and branching, which may not be desirable in performance-critical code.

    Thus, this protocol provides various alternative, precise constructors.

    That way, user who want to prioritize performance or explicitely named methods can use these alternative constructors, while still having the flexibility of `__new__` for more generic use cases.
    """
    @staticmethod
    @abstractmethod
    def of[E](*elements: E) -> FlexibleInit[E]:
        """Create the instance from a variable number of elements."""
    @staticmethod
    @abstractmethod
    def from_iter[I](iterable: Iterable[I], /) -> FlexibleInit[I]:
        """Create the instance from an `Iterable` of elements."""

@type_check_only
class FlexibleWrapper[T](FlexibleInit[T], ABC):
    @staticmethod
    @abstractmethod
    def wrap[W](wrapped: Iterable[W], /) -> FlexibleWrapper[W]:
        """Create the instance from a reference to an existing data structure corresponding to this pyochain type.

        E.g, `Vec.wrap(list)` or `Dict.wrap(dict)`.

        If you have an `Iterator`, prefer using `from_iter` instead of `Wrapper.wrap(wrapped_type)`, as it's more verbose and won't be really more efficient.

        It guarantees no-copy behavior, regardless of the mutability of the underlying data structure.

        Thus, it is the most efficient way to create a non-empty pyochain wrapper from an existing corresponding data structure.

        Warning:
            No-copy behavior means that mutable collections will be shared between the pyochain wrapper and the original data structure.

            Modifying one will affect the other.

        Args:
            wrapped (Iterable[W]): The object to wrap.

        Returns:
            FlexibleWrapper[W]: A new instance wrapping the provided `wrapped` object.

        Example:
        ```python
        from pyochain import Vec, Seq

        original_list = [1, 2, 3]
        vec = Vec.wrap(original_list)
        assert vec == Vec(1, 2, 3)
        vec[0] = 10
        assert original_list == [10, 2, 3]
        original_tuple = (1, 2, 3)
        seq = Seq.wrap(original_tuple)
        assert seq == Seq(1, 2, 3)
        ```
        """
