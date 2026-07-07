from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator, Mapping
from operator import itemgetter
from typing import TYPE_CHECKING, Any, Self, overload, override

from ..abc import PyoMutableMapping

if TYPE_CHECKING:
    from _typeshed import SupportsItems


# TODO: once in stubs, add overload to new when kwargs is passed to infer PyoCounter[str]


class PyoCounter[T](PyoMutableMapping[T, int]):  # noqa: PLW1641
    """Dict subclass for counting hashable items.

    Sometimes called a bag or multiset.
    Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    >>> c = PyoCounter("abcdeabcdabcaba")  # count elements from a string

    >>> c.most_common(3)  # three most common elements
    [('a', 5), ('b', 4), ('c', 3)]
    >>> sorted(c)  # list all unique elements
    ['a', 'b', 'c', 'd', 'e']
    >>> "".join(sorted(c.elements()))  # list elements with repetitions
    'aaaaabbbbcccdde'
    >>> sum(c.values())  # total of all counts
    15

    >>> c["a"]  # count of letter 'a'
    5
    >>> for elem in "shazam":  # update counts from an iterable
    ...     c[elem] += 1  # by adding 1 to each element's count
    >>> c["a"]  # now there are seven 'a'
    7
    >>> del c["b"]  # remove all 'b'
    >>> c["b"]  # now there are zero 'b'
    0

    >>> d = PyoCounter("simsalabim")  # make another counter
    >>> c.update(d)  # add in the second counter
    >>> c["a"]  # now there are nine 'a'
    9

    >>> c.clear()  # empty the counter
    >>> c
    PyoCounter()

    Note:  If a count is set to zero or reduced to zero, it will remain
    in the counter until the entry is deleted or the counter is cleared:

    >>> c = PyoCounter("aaabbc")
    >>> c["b"] -= 2  # reduce the count of 'b' by two
    >>> c.most_common()  # 'b' is still in, but its count is zero
    [('a', 3), ('c', 1), ('b', 0)]

    """

    _inner: dict[T, int]

    @overload
    def __init__(self, iterable: None = None, /, **kwargs: int) -> None: ...
    @overload
    def __init__(self, iterable: Mapping[T, int], /, **kwargs: int) -> None: ...
    @overload
    def __init__(self, iterable: Iterable[T], /, **kwargs: int) -> None: ...
    def __init__(
        self, iterable: Iterable[T] | Mapping[T, int] | None = None, /, **kwargs: int
    ) -> None:
        """Create a new, empty PyoCounter object.

        And if given, count elements from an input iterable.

        Or, initialize the count from another mapping of elements to their counts.

        >>> c = PyoCounter()  # a new, empty counter
        >>> c = PyoCounter("gallahad")  # a new counter from an iterable
        >>> c = PyoCounter({"a": 4, "b": 2})  # a new counter from a mapping

        """
        # Needed to emulate the behavior of stdlib PyoCounter.
        if not hasattr(self, "_inner"):
            self._inner = {}
        self.update(iterable, **kwargs)

    @override
    def __iter__(self) -> Iterator[T]:
        return iter(self._inner)

    @override
    def __len__(self) -> int:
        return len(self._inner)

    @override
    def __getitem__(self, key: T) -> int:
        try:
            return self._inner[key]
        except KeyError:
            return self.__missing__(key)

    @override
    def __setitem__(self, key: T, value: int) -> None:
        self._inner[key] = value

    @override
    def __contains__(self, key: object) -> bool:
        return key in self._inner

    def __missing__(self, key: T) -> int:
        """The count of elements not in the PyoCounter is zero.

        This is needed so that self[missing_item] does not raise `KeyError`.

        Returns:
            int: The count of the missing element, which is always zero.
        """
        return 0

    @overload
    def get(self, key: T, /) -> int | None: ...
    @overload
    def get(self, key: T, default: int, /) -> int: ...
    @overload
    def get[D](self, key: T, default: D, /) -> int | D: ...
    @override
    def get[D](self, key: T, default: D | None = None, /) -> int | D | None:
        return self._inner.get(key, default)

    @override
    def setdefault(self, key: T, default: int, /) -> int:
        return self._inner.setdefault(key, default)

    def total(self) -> int:
        """Sum of the counts.

        Returns:
            int
        """
        return sum(self.values())

    def most_common(self, n: int | None = None) -> list[tuple[T, int]]:
        """List the n most common elements and their counts from the most common to the least.

        If n is `None`, then list all element counts.

        >>> PyoCounter("abracadabra").most_common(3)
        [('a', 5), ('b', 2), ('r', 2)]

        Returns:
            list[tuple[T, int]]: A list of tuples containing the n most common elements and their counts.
        """
        match n:
            case None:
                return sorted(self.items(), key=itemgetter(1), reverse=True)
            case _:
                import heapq

                return heapq.nlargest(n, self.items(), key=itemgetter(1))

    def elements(self) -> Iterator[T]:
        """Iterator over elements repeating each as many times as its count.

        >>> c = PyoCounter("ABCABC")
        >>> sorted(c.elements())
        ['A', 'A', 'B', 'B', 'C', 'C']

        Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1

        >>> import math
        >>> prime_factors = PyoCounter({2: 2, 3: 3, 17: 1})
        >>> math.prod(prime_factors.elements())
        1836

        Note, if an element's count has been set to zero or is a negative
        number, elements() will ignore it.

        Returns:
            Iterator[T]
        """
        return itertools.chain.from_iterable(
            itertools.starmap(itertools.repeat, self.items())
        )

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

        >>> c = PyoCounter("which")
        >>> c.update("witch")  # add elements from another iterable
        >>> d = PyoCounter("watch")
        >>> c.update(d)  # add elements from another counter
        >>> c["h"]  # four 'h' in which, witch, and watch
        4

        """
        match iterable:
            case None:
                pass
            case Mapping():
                if self:
                    for elem, count in iterable.items():  # pyright: ignore[reportUnknownVariableType]
                        self[elem] = count + self.get(elem, 0)
                else:
                    # fast path when counter is empty
                    self._inner.update(iterable)  # pyright: ignore[reportUnknownArgumentType]
            case _:
                mapping_get = self._inner.get
                for elem in iterable:
                    self._inner[elem] = mapping_get(elem, 0) + 1

        if kwargs:
            self.update(kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

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

        >>> c = PyoCounter("which")
        >>> c.subtract("witch")  # subtract elements from another iterable
        >>> c.subtract(PyoCounter("watch"))  # subtract elements from another counter
        >>> c["h"]  # 2 in which, minus 1 in witch, minus 1 in watch
        0
        >>> c["w"]  # 1 in which, minus 1 in witch, minus 1 in watch
        -1

        """
        self_get = self.get
        match iterable:
            case None:
                pass
            case Mapping():
                for elem, count in iterable.items():  # pyright: ignore[reportUnknownVariableType]
                    self[elem] = self_get(elem, 0) - count
            case _:
                for elem in iterable:
                    self[elem] = self_get(elem, 0) - 1
        if kwargs:
            self.subtract(kwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def copy(self) -> Self:
        """Return a shallow copy."""
        return self.__class__(self)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[dict[T, int]]]:
        return self.__class__, (dict(self),)

    @override
    def __delitem__(self, elem: T) -> None:
        """Like dict.__delitem__() but does not raise KeyError for missing values."""
        if elem in self:
            self._inner.__delitem__(elem)

    @override
    def __repr__(self) -> str:
        if not self:
            return f"{self.__class__.__name__}()"
        try:
            # dict() preserves the ordering returned by most_common()
            d = dict(self.most_common())
        except TypeError:
            # handle case where values are not orderable
            d = dict(self)
        return f"{self.__class__.__name__}({d!r})"

    def __add__[S](self, other: PyoCounter[S]) -> PyoCounter[T | S]:
        """Add counts from two counters.

        >>> PyoCounter("abbb") + PyoCounter("bcc")
        PyoCounter({'b': 4, 'c': 2, 'a': 1})

        Returns:
            PyoCounter[T | S]: A new counter with the added counts.
        """
        result = PyoCounter[T | S]()
        for elem, count in self.items():
            newcount = count + other[elem]  # pyright: ignore[reportArgumentType]
            if newcount > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and count > 0:
                result[elem] = count
        return result

    def __sub__(self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Subtract count, but keep only results with positive counts.

        >>> PyoCounter("abbbc") - PyoCounter("bccd")
        PyoCounter({'b': 2, 'a': 1})

        Returns:
            PyoCounter[T]: A new counter with the subtracted counts, keeping only positive counts.
        """
        result = PyoCounter[T]()
        for elem, count in self.items():
            newcount = count - other[elem]
            if newcount > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and count < 0:
                result[elem] = 0 - count
        return result

    def __or__[S](self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Union is the maximum of value in either of the input counters.

        >>> PyoCounter("abbb") | PyoCounter("bcc")
        PyoCounter({'b': 3, 'c': 2, 'a': 1})

        Returns:
            PyoCounter[T]: A new counter with the union of counts.
        """
        result = PyoCounter[T]()
        for elem, count in self.items():
            other_count = other[elem]
            newcount = max(count, other_count)
            if newcount > 0:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and count > 0:
                result[elem] = count
        return result

    def __and__(self, other: PyoCounter[T]) -> PyoCounter[T]:
        """Intersection is the minimum of corresponding counts.

        >>> PyoCounter("abbb") & PyoCounter("bcc")
        PyoCounter({'b': 1})

        Returns:
            PyoCounter[T]: A new counter with the intersection of counts.
        """
        result = PyoCounter[T]()
        for elem, count in self.items():
            other_count = other[elem]
            newcount = min(other_count, count)
            if newcount > 0:
                result[elem] = newcount
        return result

    def __pos__(self) -> PyoCounter[T]:
        """Adds an empty counter, effectively stripping negative and zero counts.

        Returns:
            PyoCounter[T]: A new counter with only positive counts.
        """
        result = PyoCounter[T]()
        for elem, count in self.items():
            if count > 0:
                result[elem] = count
        return result

    def __neg__(self) -> PyoCounter[T]:
        """Subtracts from an empty counter.

        Strips positive and zero counts, and flips the sign on negative counts.

        Returns:
            PyoCounter[T]: A new counter.
        """
        result = PyoCounter[T]()
        for elem, count in self.items():
            if count < 0:
                result[elem] = 0 - count
        return result

    def __iadd__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace add from another counter, keeping only positive counts.

        >>> c = PyoCounter("abbb")
        >>> c += PyoCounter("bcc")
        >>> c
        PyoCounter({'b': 4, 'c': 2, 'a': 1})

        Returns:
            Self: The updated counter with the added counts.
        """
        for elem, count in other.items():
            self[elem] += count
        return self._keep_positive()

    def __isub__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace subtract counter, but keep only results with positive counts.

        >>> c = PyoCounter("abbbc")
        >>> c -= PyoCounter("bccd")
        >>> c
        PyoCounter({'b': 2, 'a': 1})

        Returns:
            Self: The updated counter with the subtracted counts.
        """
        for elem, count in other.items():
            self[elem] -= count
        return self._keep_positive()

    def __ior__(self, other: SupportsItems[T, int]) -> Self:
        """Inplace union is the maximum of value from either counter.

        >>> c = PyoCounter("abbb")
        >>> c |= PyoCounter("bcc")
        >>> c
        PyoCounter({'b': 3, 'c': 2, 'a': 1})

        Returns:
            Self: The updated counter with the union of counts.

        """
        for elem, other_count in other.items():
            count = self[elem]
            if other_count > count:
                self[elem] = other_count
        return self._keep_positive()

    def __iand__(self, other: Mapping[T, int]) -> Self:
        """Inplace intersection is the minimum of corresponding counts.

        >>> c = PyoCounter("abbb")
        >>> c &= PyoCounter("bcc")
        >>> c
        PyoCounter({'b': 1})

        Returns:
            Self: The updated counter with the intersection of counts.
        """
        for elem, count in self.items():
            other_count = other[elem]
            if other_count < count:
                self[elem] = other_count
        return self._keep_positive()

    @override
    def __eq__(self, other: object) -> bool:
        """True if all counts agree. Missing counts are treated as zero.

        Returns:
            bool
        """
        match other:
            case PyoCounter():
                return all(self[e] == other[e] for c in (self, other) for e in c)  # pyright: ignore[reportUnknownVariableType]
            case dict():
                return self._inner == other
            case _:
                return NotImplemented

    @override
    def __ne__(self, other: object) -> bool:
        """True if any counts disagree. Missing counts are treated as zero.

        Returns:
            bool
        """
        if not isinstance(other, PyoCounter):
            return NotImplemented
        return not self == other

    def __le__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a subset of those in other.

        Returns:
            bool
        """
        return all(self[e] <= other[e] for c in (self, other) for e in c)

    def __lt__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a proper subset of those in other.

        Returns:
            bool
        """
        return self <= other and self != other

    def __ge__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a superset of those in other.

        Returns:
            bool
        """
        return all(self[e] >= other[e] for c in (self, other) for e in c)

    def __gt__(self, other: PyoCounter[Any]) -> bool:
        """True if all counts in self are a proper superset of those in other.

        Returns:
            bool
        """
        return self >= other and self != other

    def __xor__[S](self, other: PyoCounter[S]) -> PyoCounter[T | S]:
        """Symmetric difference. Absolute value of count differences.

        The symmetric difference p ^ q is equivalent to:

            (p - q) | (q - p).

        For each element, symmetric difference gives the same result as:

            max(p[elem], q[elem]) - min(p[elem], q[elem])

        >>> PyoCounter(a=5, b=3, c=2, d=2) ^ PyoCounter(a=1, b=3, c=5, e=1)
        PyoCounter({'a': 4, 'c': 3, 'd': 2, 'e': 1})

        Returns:
            PyoCounter[T | S]: A new counter with the symmetric difference of counts.

        """
        result = PyoCounter[T | S]()
        for elem, count in self.items():
            newcount = abs(count - other[elem])  # pyright: ignore[reportArgumentType]
            if newcount:
                result[elem] = newcount
        for elem, count in other.items():
            if elem not in self and count:
                result[elem] = abs(count)
        return result

    def __ixor__(self, other: PyoCounter[T]) -> Self:
        """Inplace symmetric difference. Absolute value of count differences.

        >>> c = PyoCounter(a=5, b=3, c=2, d=2)
        >>> c ^= PyoCounter(a=1, b=3, c=5, e=1)
        >>> c
        PyoCounter({'a': 4, 'c': 3, 'd': 2, 'e': 1})

        Returns:
            Self: The updated counter with the symmetric difference of counts.
        """
        for elem, count in self.items():
            self[elem] = abs(count - other[elem])
        for elem, count in other.items():
            if elem not in self:
                self[elem] = abs(count)
        return self._keep_positive()

    def _keep_positive(self) -> Self:
        """Internal method to strip elements with a negative or zero count.

        Returns:
            Self: The updated counter with only positive counts.
        """
        nonpositive = [elem for elem, count in self.items() if not count > 0]
        for elem in nonpositive:
            del self[elem]
        return self
