from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Self, final, overload, override

from _typeshed import SupportsItems, SupportsKeysAndGetItem

from pyochain import Vec
from pyochain.abc import PyoIterator, PyoMutableMapping, PyoReversible

@final
class PyoCounter[T](PyoMutableMapping[T, int], PyoReversible[T]):
    """Dict subclass for counting hashable items.

    Sometimes called a bag or multiset.
    Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    ```python
    from pyochain.collections import PyoCounter

    c = PyoCounter("abcdeabcdabcaba")  # count elements from a string

    # three most common elements
    assert c.most_common(3) == [("a", 5), ("b", 4), ("c", 3)]

    # list all unique elements
    assert c.iter().sort() == ["a", "b", "c", "d", "e"]

    # list elements with repetitions
    joined = c.elements().iter().sort().iter().join("")
    assert joined == "aaaaabbbbcccdde"

    # total of all counts
    assert c.values().iter().sum() == 15
    # count of letter 'a'
    assert c["a"] == 5
    # update counts from an iterable
    for elem in "shazam":
        # by adding 1 to each element's count
        c[elem] += 1
    # now there are seven 'a'
    assert c["a"] == 7

    # remove all 'b'
    del c["b"]
    # now there are zero 'b'
    assert c["b"] == 0

    # make another counter
    d = PyoCounter("simsalabim")
    # add in the second counter
    c.update(d)
    # now there are nine 'a'
    assert c["a"] == 9

    c.clear()  # empty the counter
    assert c == PyoCounter()
    ```

    Note:  If a count is set to zero or reduced to zero, it will remain
    in the counter until the entry is deleted or the counter is cleared:
    ```python
    c = PyoCounter("aaabbc")
    c["b"] -= 2  # reduce the count of 'b' by two
    # 'b' is still in, but its count is zero
    assert c.most_common() == [("a", 3), ("c", 1), ("b", 0)]
    ```
    If given, count elements from an input iterable.

    Or, initialize the count from another mapping of elements to their counts.
    ```python
    c = PyoCounter()  # a new, empty counter
    assert c.is_empty()
    c = PyoCounter("gallahad")  # a new counter from an iterable
    assert c["a"] == 3
    c = PyoCounter({"a": 4, "b": 2})  # a new counter from a mapping
    assert c["b"] == 2
    ```

    """
    @staticmethod
    def wrap[S](data: dict[S, int]) -> PyoCounter[S]: ...
    @overload
    def __new__(cls, /) -> Self: ...
    @overload
    def __new__(
        cls: type[PyoCounter[str]], iterable: None = None, /, **kwargs: int
    ) -> PyoCounter[str]: ...
    @overload
    def __new__(cls, mapping: SupportsKeysAndGetItem[T, int], /) -> Self: ...
    @overload
    def __new__(cls, iterable: Iterable[T], /) -> Self: ...
    @override
    def __iter__(self) -> Iterator[T]: ...
    @override
    def __len__(self) -> int: ...
    @override
    def __getitem__(self, key: T) -> int: ...
    @override
    def __setitem__(self, key: T, value: int) -> None: ...
    @override
    def __contains__(self, key: object) -> bool: ...
    def __missing__(self, key: T) -> int:
        """The count of elements not in the PyoCounter is zero.

        This is needed so that self[missing_item] does not raise `KeyError`.

        Args:
            key (T): The missing element to look up.

        Returns:
            int: The count of the missing element, which is always zero.
        """

    @override
    def __reversed__(self) -> Iterator[T]: ...
    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[T, int]]]: ...
    @override
    def __delitem__(self, elem: T) -> None:
        """Like dict.__delitem__() but does not raise KeyError for missing values."""

    def __add__[S](self, other: PyoCounter[S]) -> PyoCounter[T | S]:
        """Add counts from two counters.

        ```python
        from pyochain.collections import PyoCounter

        added = PyoCounter("abbb") + PyoCounter("bcc")
        assert added == PyoCounter({"b": 4, "c": 2, "a": 1})
        ```

        Args:
            other (PyoCounter[S]): Another counter to add counts from.

        Returns:
            PyoCounter[T | S]: A new counter with the added counts.
        """

    def __sub__(self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Subtract count, but keep only results with positive counts.

        ```python
        from pyochain.collections import PyoCounter

        subtracted = PyoCounter("abbbc") - PyoCounter("bccd")
        assert subtracted == PyoCounter({"b": 2, "a": 1})
        ```

        Args:
            other (PyoCounter[T]): Another counter to subtract counts from.

        Returns:
            PyoCounter[T]: A new counter with the subtracted counts, keeping only positive counts.
        """

    def __or__[S](self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Union is the maximum of value in either of the input counters.

        ```python
        from pyochain.collections import PyoCounter

        union = PyoCounter("abbb") | PyoCounter("bcc")
        assert union == PyoCounter({"b": 3, "c": 2, "a": 1})
        ```


        Args:
            other (PyoCounter[T]): Another counter to take the union with.

        Returns:
            PyoCounter[T]: A new counter with the union of counts.
        """

    def __and__(self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Intersection is the minimum of corresponding counts.

        ```python
        from pyochain.collections import PyoCounter

        union = PyoCounter("abbb") & PyoCounter("bcc")
        assert union == PyoCounter({"b": 1})
        ```

        Args:
            other (PyoCounter[T]): Another counter to take the intersection with.

        Returns:
            PyoCounter[T]: A new counter with the intersection of counts.
        """

    def __pos__(self) -> PyoCounter[T]:
        """Adds an empty counter, effectively stripping negative and zero counts.

        Returns:
            PyoCounter[T]: A new counter with only positive counts.
        """

    def __neg__(self) -> PyoCounter[T]:
        """Subtracts from an empty counter.

        Strips positive and zero counts, and flips the sign on negative counts.

        Returns:
            PyoCounter[T]: A new counter.
        """

    def __iadd__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace add from another counter, keeping only positive counts.

        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("abbb")
        c += PyoCounter("bcc")
        assert c == PyoCounter({"b": 4, "c": 2, "a": 1})
        ```

        Args:
            other (SupportsItems[T, int]): Another counter or mapping to add counts from.

        Returns:
            Self: The updated counter with the added counts.
        """

    def __isub__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace subtract counter, but keep only results with positive counts.

        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("abbbc")
        c -= PyoCounter("bccd")
        assert c == PyoCounter({"b": 2, "a": 1})
        ```

        Args:
            other (SupportsItems[T, int]): Another counter or mapping to subtract counts from.

        Returns:
            Self: The updated counter with the subtracted counts.
        """

    def __ior__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace union is the maximum of value from either counter.

        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("abbb")
        c |= PyoCounter("bcc")
        assert c == PyoCounter({"b": 3, "c": 2, "a": 1})
        ```

        Args:
            other (SupportsItems[T, int]): Another counter or mapping to take the union with.

        Returns:
            Self: The updated counter with the union of counts.

        """

    def __iand__(self, other: Mapping[T, int]) -> Self:
        """Inplace intersection is the minimum of corresponding counts.

        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("abbb")
        c &= PyoCounter("bcc")
        assert c == PyoCounter({"b": 1})
        ```

        Args:
            other (Mapping[T, int]): Another counter or mapping to take the intersection with.

        Returns:
            Self: The updated counter with the intersection of counts.
        """

    @override
    def __eq__(self, other: object) -> bool:
        """True if all counts agree. Missing counts are treated as zero.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if all counts agree, False otherwise. If `other` is not a PyoCounter or dict, returns NotImplemented.
        """

    @override
    def __ne__(self, other: object) -> bool:
        """True if any counts disagree. Missing counts are treated as zero.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True if any counts disagree, False otherwise. If `other` is not a PyoCounter or dict, returns NotImplemented.
        """

    def __le__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a subset of those in other.

        Args:
            other (PyoCounter[Any]): The counter to compare with.

        Returns:
            bool: True if all counts in self are a subset of those in other, False otherwise.
        """

    def __lt__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a proper subset of those in other.

        Args:
            other (PyoCounter[Any]): The counter to compare with.

        Returns:
            bool: True if all counts in self are a proper subset of those in other, False otherwise.
        """

    def __ge__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a superset of those in other.

        Args:
            other (PyoCounter[Any]): The counter to compare with.

        Returns:
            bool: True if all counts in self are a superset of those in other, False otherwise.
        """

    def __gt__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a proper superset of those in other.

        Args:
            other (PyoCounter[Any]): The counter to compare with.

        Returns:
            bool: True if all counts in self are a proper superset of those in other, False otherwise.
        """

    def __xor__[S](self, other: PyoCounter[S]) -> PyoCounter[T | S]:
        """Symmetric difference. Absolute value of count differences.

        The symmetric difference p ^ q is equivalent to:

            (p - q) | (q - p).

        For each element, symmetric difference gives the same result as:

            max(p[elem], q[elem]) - min(p[elem], q[elem])


        Args:
            other (PyoCounter[S]): The counter to compare with.

        Returns:
            PyoCounter[T | S]: A new counter with the symmetric difference of counts.

        Example:
            ```python
            from pyochain.collections import PyoCounter

            symmetric_diff = PyoCounter(a=5, b=3, c=2, d=2) ^ PyoCounter(
                a=1, b=3, c=5, e=1
            )
            assert symmetric_diff == PyoCounter({"a": 4, "c": 3, "d": 2, "e": 1})
            ```
        """

    def __ixor__(self, other: PyoCounter[T]) -> Self:
        """Inplace symmetric difference. Absolute value of count differences.

        Args:
            other (PyoCounter[T]): The counter to compare with.

        Returns:
            Self: The updated counter with the symmetric difference of counts.

        Example:
            ```python
            from pyochain.collections import PyoCounter

            c = PyoCounter(a=5, b=3, c=2, d=2)
            c ^= PyoCounter(a=1, b=3, c=5, e=1)
            assert c == PyoCounter({"a": 4, "c": 3, "d": 2, "e": 1})
            ```
        """

    @overload
    def get(self, key: T, /) -> int | None: ...
    @overload
    def get(self, key: T, default: int, /) -> int: ...
    @overload
    def get[D](self, key: T, default: D, /) -> int | D: ...
    @override
    def get[D](self, key: T, default: D | None = None, /) -> int | D | None: ...
    @override
    def setdefault(self, key: T, default: int, /) -> int: ...
    def total(self) -> int:
        """Sum of the counts.

        Returns:
            int: The sum of all counts in the PyoCounter.
        """

    def most_common(self, n: int | None = None) -> Vec[tuple[T, int]]:
        """List the n most common elements and their counts from the most common to the least.

        ```python
        from pyochain.collections import PyoCounter

        most_commons = PyoCounter("abracadabra").most_common(3)
        assert most_commons == [("a", 5), ("b", 2), ("r", 2)]
        ```

        Args:
            n (int | None): The number of most common elements to return. If `None`, return all elements.

        Returns:
            Vec[tuple[T, int]]: A list of tuples containing the n most common elements and their counts.
        """

    def elements(self) -> PyoIterator[T]:
        """`Iterator` over elements repeating each as many times as its count.

        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("ABCABC")
        assert c.elements().sort() == ["A", "A", "B", "B", "C", "C"]
        ```

        Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1

        ```python
        import math

        prime_factors = PyoCounter({2: 2, 3: 3, 17: 1})
        assert math.prod(prime_factors.elements()) == 1836
        ```

        Note, if an element's count has been set to zero or is a negative
        number, elements() will ignore it.

        Returns:
            PyoIterator[T]: An iterator over elements repeating each as many times as its count.
        """

    @overload
    def update(self, iterable: None = None, /, **kwargs: int) -> None: ...
    @overload
    def update(self, iterable: Mapping[T, int], /, **kwargs: int) -> None: ...
    @overload
    def update(self, iterable: Iterable[T], /, **kwargs: int) -> None: ...
    @override
    def update(
        self, iterable: Mapping[T, int] | Iterable[T] | None = None, /, **kwargs: int
    ) -> None:
        """Like dict.update() but add counts instead of replacing them.

        Source can be an iterable, a dictionary, or another PyoCounter instance.

        Note:
            The regular dict.update() operation makes no sense here because the
            replace behavior results in some of the original untouched counts
            being mixed-in with all of the other counts for a mismash that
            doesn't have a straight-forward interpretation in most counting
            contexts.
            Instead, we implement straight-addition.
            Both the inputs and outputs are allowed to contain zero and negative counts.
        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("which")
        c.update("witch")  # add elements from another iterable
        d = PyoCounter("watch")
        c.update(d)  # add elements from another counter
        # four 'h' in which, witch, and watch
        assert c["h"] == 4
        ```

        """

    @overload
    def subtract(self, iterable: None = None, /, **kwargs: int) -> None: ...
    @overload
    def subtract(self, mapping: Mapping[T, int], /, **kwargs: int) -> None: ...
    @overload
    def subtract(self, iterable: Iterable[T], /, **kwargs: int) -> None: ...
    def subtract(
        self, iterable: Mapping[T, int] | Iterable[T] | None = None, /, **kwargs: int
    ) -> None:
        """Like dict.update() but subtracts counts instead of replacing them.

        Counts can be reduced below zero.  Both the inputs and outputs are
        allowed to contain zero and negative counts.

        Source can be an iterable, a dictionary, or another PyoCounter instance.


        ```python
        from pyochain.collections import PyoCounter

        c = PyoCounter("which")
        c.subtract("witch")  # subtract elements from another iterable
        c.subtract(PyoCounter("watch"))  # subtract elements from another counter
        # 2 in which, minus 1 in witch, minus 1 in watch
        assert c["h"] == 0
        # 1 in which, minus 1 in witch, minus 1 in watch
        assert c["w"] == -1
        ```

        """

    def copy(self) -> Self:
        """Return a shallow copy."""
