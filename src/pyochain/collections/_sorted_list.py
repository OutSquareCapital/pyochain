# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0
from __future__ import annotations

from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Iterable, Iterator, MutableSequence, Sequence
from functools import reduce
from itertools import chain, repeat
from math import log2
from operator import add, eq, ge, gt, iadd, le, lt, ne
from reprlib import recursive_repr
from textwrap import dedent
from typing import TYPE_CHECKING, Final, Self, overload, override

if TYPE_CHECKING:
    from types import NotImplementedType

    from _typeshed import SupportsRichComparison

type KeyFunc[T, OT: SupportsRichComparison] = Callable[[T], OT]


class SortedList[T: SupportsRichComparison](MutableSequence[T]):  # noqa: PLW1641
    """Sorted list is a sorted mutable sequence.

    Sorted list values are maintained in sorted order.

    Sorted list values must be comparable. The total ordering of values must
    not change while they are stored in the sorted list.

    Methods for adding values:

    * :func:`SortedList.add`
    * :func:`SortedList.update`
    * :func:`SortedList.__add__`
    * :func:`SortedList.__iadd__`
    * :func:`SortedList.__mul__`
    * :func:`SortedList.__imul__`

    Methods for removing values:

    * :func:`SortedList.clear`
    * :func:`SortedList.discard`
    * :func:`SortedList.remove`
    * :func:`SortedList.pop`
    * :func:`SortedList.__delitem__`

    Methods for looking up values:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.count`
    * :func:`SortedList.index`
    * :func:`SortedList.__contains__`
    * :func:`SortedList.__getitem__`

    Methods for iterating values:

    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList.__iter__`
    * :func:`SortedList.__reversed__`

    Methods for miscellany:

    * :func:`SortedList.copy`
    * :func:`SortedList.__len__`
    * :func:`SortedList.__repr__`
    * :func:`SortedList._check`
    * :func:`SortedList.reset`

    Sorted lists use lexicographical ordering semantics when compared to other
    sequences.

    Some methods of mutable sequences are not supported and will raise
    not-implemented error.

    """

    DEFAULT_LOAD_FACTOR: Final[int] = 1000

    def __init__(self, iterable: Iterable[T] | None = None) -> None:
        """Initialize sorted list instance.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted list.

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList()
        >>> sl
        SortedList([])
        >>> sl = SortedList([3, 1, 2, 5, 4])
        >>> sl
        SortedList([1, 2, 3, 4, 5])

        :param iterable: initial values (optional)

        """
        self._len: int = 0
        self._load: int = self.DEFAULT_LOAD_FACTOR
        self._lists: list[list[T]] = []
        self._maxes: list[T] = []
        self._index: list[int] = []
        self._offset: int = 0

        if iterable is not None:
            self.update(iterable)

    def reset(self, load: int) -> None:
        """Reset sorted list load factor.

        The `load` specifies the load-factor of the list. The default load
        factor of 1000 works well for lists from tens to tens-of-millions of
        values. Good practice is to use a value that is the cube root of the
        list size. With billions of elements, the best load factor depends on
        your usage. It's best to leave the load factor at the default until you
        start benchmarking.

        See :doc:`implementation` and :doc:`performance-scale` for more
        information.

        Runtime complexity: `O(n)`

        :param int load: load-factor for sorted list sublists

        """
        values: list[T] = reduce(iadd, self._lists, [])
        self.clear()
        self._load = load
        self.update(values)

    @override
    def clear(self) -> None:
        """Remove all values from sorted list.

        Runtime complexity: `O(n)`

        """
        self._len = 0
        del self._lists[:]
        del self._maxes[:]
        del self._index[:]
        self._offset = 0

    def add(self, value: T) -> None:
        """Add `value` to sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList()
        >>> sl.add(3)
        >>> sl.add(1)
        >>> sl.add(2)
        >>> sl
        SortedList([1, 2, 3])

        :param value: value to add to sorted list

        """
        lists = self._lists
        maxes = self._maxes

        if maxes:
            pos = bisect_right(maxes, value)

            if pos == len(maxes):
                pos -= 1
                lists[pos].append(value)
                maxes[pos] = value
            else:
                insort(lists[pos], value)

            self._expand(pos)
        else:
            lists.append([value])
            maxes.append(value)

        self._len += 1

    def _expand(self, pos: int) -> None:
        """Split sublists with length greater than double the load-factor.

        Updates the index when the sublist length is less than double the load
        level. This requires incrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        """
        load = self._load
        lists = self._lists
        index = self._index

        if len(lists[pos]) > (load << 1):
            maxes = self._maxes

            lists_pos = lists[pos]
            half = lists_pos[load:]
            del lists_pos[load:]
            maxes[pos] = lists_pos[-1]

            lists.insert(pos + 1, half)
            maxes.insert(pos + 1, half[-1])

            del index[:]
        elif index:
            child = self._offset + pos
            while child:
                index[child] += 1
                child = (child - 1) >> 1
            index[0] += 1

    def update(self, iterable: Iterable[T]) -> None:
        """Update sorted list by adding all values from `iterable`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> sl = SortedList()
        >>> sl.update([3, 1, 2])
        >>> sl
        SortedList([1, 2, 3])

        :param iterable: iterable of values to add

        """
        lists = self._lists
        maxes = self._maxes
        values: list[T] = sorted(iterable)

        if maxes:
            if len(values) * 4 >= self._len:
                lists.append(values)
                values = reduce(iadd, lists, [])
                values.sort()
                self.clear()
            else:
                add_ = self.add
                for val in values:
                    add_(val)
                return

        load = self._load
        lists.extend(values[pos : (pos + load)] for pos in range(0, len(values), load))
        maxes.extend(sublist[-1] for sublist in lists)
        self._len = len(values)
        del self._index[:]

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted list.

        ``sl.__contains__(value)`` <==> ``value in sl``

        Runtime complexity: `O(log(n))`

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> 3 in sl
        True

        :param value: search for value in sorted list
        :return: true if `value` in sorted list

        """
        maxes = self._maxes

        if not maxes:
            return False

        pos = bisect_left(maxes, value)

        if pos == len(maxes):
            return False

        lists = self._lists
        idx = bisect_left(lists[pos], value)

        return lists[pos][idx] == value

    def discard(self, value: T) -> None:
        """Remove `value` from sorted list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.discard(5)
        >>> sl.discard(0)
        >>> sl == [1, 2, 3, 4]
        True

        :param value: `value` to discard from sorted list

        """
        maxes = self._maxes

        if not maxes:
            return

        pos = bisect_left(maxes, value)

        if pos == len(maxes):
            return

        lists = self._lists
        idx = bisect_left(lists[pos], value)

        if lists[pos][idx] == value:
            self._delete(pos, idx)

    @override
    def remove(self, value: T) -> None:
        """Remove `value` from sorted list; `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 3, 4, 5])
        >>> sl.remove(5)
        >>> sl == [1, 2, 3, 4]
        True
        >>> sl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list

        :param value: `value` to remove from sorted list

        Raises:
            ValueError: if `value` is not in sorted list

        """
        maxes = self._maxes

        if not maxes:
            msg = f"{value!r} not in list"
            raise ValueError(msg)

        pos = bisect_left(maxes, value)

        if pos == len(maxes):
            msg = f"{value!r} not in list"
            raise ValueError(msg)

        lists = self._lists
        idx = bisect_left(lists[pos], value)

        if lists[pos][idx] == value:
            self._delete(pos, idx)
        else:
            msg = f"{value!r} not in list"
            raise ValueError(msg)

    def _delete(self, pos: int, idx: int) -> None:
        """Delete value at the given `(pos, idx)`.

        Combines lists that are less than half the load level.

        Updates the index when the sublist length is more than half the load
        level. This requires decrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        :param int pos: lists index
        :param int idx: sublist index

        """
        lists = self._lists
        maxes = self._maxes
        index = self._index

        lists_pos = lists[pos]

        del lists_pos[idx]
        self._len -= 1

        len_lists_pos = len(lists_pos)

        if len_lists_pos > (self._load >> 1):
            maxes[pos] = lists_pos[-1]

            if index:
                child = self._offset + pos
                while child > 0:
                    index[child] -= 1
                    child = (child - 1) >> 1
                index[0] -= 1
        elif len(lists) > 1:
            if not pos:
                pos += 1

            prev = pos - 1
            lists[prev].extend(lists[pos])
            maxes[prev] = lists[prev][-1]

            del lists[pos]
            del maxes[pos]
            del index[:]

            self._expand(prev)
        elif len_lists_pos:
            maxes[pos] = lists_pos[-1]
        else:
            del lists[pos]
            del maxes[pos]
            del index[:]

    def _loc(self, pos: int, idx: int) -> int:
        """Convert an index pair (lists index, sublist index) into a single index number.

        This number corresponds to the position of the value in the
        sorted list.

        Many queries require the index be built. Details of the index are
        described in ``SortedList._build_index``.

        Indexing requires traversing the tree from a leaf node to the root. The
        parent of each node is easily computable at ``(pos - 1) // 2``.

        Left-child nodes are always at odd indices and right-child nodes are
        always at even indices.

        When traversing up from a right-child node, increment the total by the
        left-child node.

        The final index is the sum from traversal and the index in the sublist.

        For example, using the index from ``SortedList._build_index``::

            _index = 14 5 9 3 2 4 5
            _offset = 3

        Tree::

                 14
              5      9
            3   2  4   5

        Converting an index pair (2, 3) into a single index involves iterating
        like so:

        1. Starting at the leaf node: offset + alpha = 3 + 2 = 5. We identify
           the node as a left-child node. At such nodes, we simply traverse to
           the parent.

        2. At node 9, position 2, we recognize the node as a right-child node
           and accumulate the left-child in our total. Total is now 5 and we
           traverse to the parent at position 0.

        3. Iteration ends at the root.

        The index is then the sum of the total and sublist index: 5 + 3 = 8.

        :param int pos: lists index
        :param int idx: sublist index

        Returns:
            index in sorted list

        """
        if not pos:
            return idx

        index = self._index

        if not index:
            self._build_index()

        total = 0

        # Increment pos to point in the index to len(self._lists[pos]).

        pos += self._offset

        # Iterate until reaching the root of the index tree at pos = 0.

        while pos:
            # Right-child nodes are at odd indices. At such indices
            # account the total below the left child node.

            if not pos & 1:
                total += index[pos - 1]

            # Advance pos to the parent node.

            pos = (pos - 1) >> 1

        return total + idx

    def _pos(self, idx: int) -> tuple[int, int]:
        """Convert an index into an index pair (lists index, sublist index).

        This pair can be used to access the corresponding lists position.

        Many queries require the index be built. Details of the index are
        described in ``SortedList._build_index``.

        Indexing requires traversing the tree to a leaf node. Each node has two
        children which are easily computable. Given an index, pos, the
        left-child is at ``pos * 2 + 1`` and the right-child is at ``pos * 2 +
        2``.

        When the index is less than the left-child, traversal moves to the
        left sub-tree. Otherwise, the index is decremented by the left-child
        and traversal moves to the right sub-tree.

        At a child node, the indexing pair is computed from the relative
        position of the child node as compared with the offset and the remaining
        index.

        For example, using the index from ``SortedList._build_index``::

            _index = 14 5 9 3 2 4 5
            _offset = 3

        Tree::

                 14
              5      9
            3   2  4   5

        Indexing position 8 involves iterating like so:

        1. Starting at the root, position 0, 8 is compared with the left-child
           node (5) which it is greater than. When greater the index is
           decremented and the position is updated to the right child node.

        2. At node 9 with index 3, we again compare the index to the left-child
           node with value 4. Because the index is the less than the left-child
           node, we simply traverse to the left.

        3. At node 4 with index 3, we recognize that we are at a leaf node and
           stop iterating.

        4. To compute the sublist index, we subtract the offset from the index
           of the leaf node: 5 - 3 = 2. To compute the index in the sublist, we
           simply use the index remaining from iteration. In this case, 3.

        The final index pair from our example is (2, 3) which corresponds to
        index 8 in the sorted list.

        :param int idx: index in sorted list

        Raises:
            IndexError: if `idx` is out of range

        Returns:
            (lists index, sublist index) pair

        """
        if idx < 0:
            last_len = len(self._lists[-1])

            if (-idx) <= last_len:
                return len(self._lists) - 1, last_len + idx

            idx += self._len

            if idx < 0:
                msg = "list index out of range"
                raise IndexError(msg)
        elif idx >= self._len:
            msg = "list index out of range"
            raise IndexError(msg)

        if idx < len(self._lists[0]):
            return 0, idx

        index = self._index

        if not index:
            self._build_index()

        pos = 0
        child = 1
        len_index = len(index)

        while child < len_index:
            index_child = index[child]

            if idx < index_child:
                pos = child
            else:
                idx -= index_child
                pos = child + 1

            child = (pos << 1) + 1

        return (pos - self._offset, idx)

    def _build_index(self) -> None:
        """Build a positional index for indexing the sorted list.

        Indexes are represented as binary trees in a dense array notation
        similar to a binary heap.

        For example, given a lists representation storing integers::

            0: [1, 2, 3]
            1: [4, 5]
            2: [6, 7, 8, 9]
            3: [10, 11, 12, 13, 14]

        The first transformation maps the sub-lists by their length. The
        first row of the index is the length of the sub-lists::

            0: [3, 2, 4, 5]

        Each row after that is the sum of consecutive pairs of the previous
        row::

            1: [5, 9]
            2: [14]

        Finally, the index is built by concatenating these lists together::

            _index = [14, 5, 9, 3, 2, 4, 5]

        An offset storing the start of the first row is also stored::

            _offset = 3

        When built, the index can be used for efficient indexing into the list.
        See the comment and notes on ``SortedList._pos`` for details.

        """
        row0 = list(map(len, self._lists))

        if len(row0) == 1:
            self._index[:] = row0
            self._offset = 0
            return

        head = iter(row0)
        tail = iter(head)
        row1 = list(map(add, head, tail))

        if len(row0) & 1:
            row1.append(row0[-1])

        if len(row1) == 1:
            self._index[:] = row1 + row0
            self._offset = 1
            return

        size = 2 ** (int(log2(len(row1) - 1)) + 1)
        row1.extend(repeat(0, size - len(row1)))
        tree = [row0, row1]

        while len(tree[-1]) > 1:
            head = iter(tree[-1])
            tail = iter(head)
            row = list(map(add, head, tail))
            tree.append(row)

        _ = reduce(iadd, reversed(tree), self._index)
        self._offset = size * 2 - 1

    @override
    def __delitem__(self, index: int | slice) -> None:
        """Remove value at `index` from sorted list.

        ``sl.__delitem__(index)`` <==> ``del sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList("abcde")
        >>> del sl[2]
        >>> sl
        SortedList(['a', 'b', 'd', 'e'])
        >>> del sl[:2]
        >>> sl
        SortedList(['d', 'e'])

        :param index: integer or slice for indexing
        :raises IndexError: if index out of range

        """
        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)

            if step == 1 and start < stop:
                if start == 0 and stop == self._len:
                    return self.clear()
                if self._len <= 8 * (stop - start):
                    values = self.__getitem__(slice(None, start))
                    if stop < self._len:
                        values += self.__getitem__(slice(stop, None))
                    self.clear()
                    return self.update(values)

            indices = range(start, stop, step)

            # Delete items from greatest index to least so
            # that the indices remain valid throughout iteration.

            if step > 0:
                indices = reversed(indices)

            pos_, delete = self._pos, self._delete

            for index in indices:
                pos, idx = pos_(index)
                delete(pos, idx)
        else:
            pos, idx = self._pos(index)
            self._delete(pos, idx)
        return None

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...
    @override
    def __getitem__(self, index: int | slice) -> T | list[T]:  # noqa: C901, PLR0911, PLR0912, PLR0914
        """Lookup value at `index` in sorted list.

        ``sl.__getitem__(index)`` <==> ``sl[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList("abcde")
        >>> sl[1]
        'b'
        >>> sl[-1]
        'e'
        >>> sl[2:5]
        ['c', 'd', 'e']

        :param index: integer or slice for indexing

        Returns:
            value or list of values

        Raises:
            IndexError: if index out of range

        """
        lists = self._lists

        if isinstance(index, slice):
            start, stop, step = index.indices(self._len)

            if step == 1 and start < stop:
                # Whole slice optimization: start to stop slices the whole
                # sorted list.

                if start == 0 and stop == self._len:
                    return reduce(iadd, self._lists, [])

                start_pos, start_idx = self._pos(start)
                start_list = lists[start_pos]
                stop_idx = start_idx + stop - start

                # Small slice optimization: start index and stop index are
                # within the start list.

                if len(start_list) >= stop_idx:
                    return start_list[start_idx:stop_idx]

                if stop == self._len:
                    stop_pos = len(lists) - 1
                    stop_idx = len(lists[stop_pos])
                else:
                    stop_pos, stop_idx = self._pos(stop)

                prefix = lists[start_pos][start_idx:]
                middle = lists[(start_pos + 1) : stop_pos]
                result = reduce(iadd, middle, prefix)
                result += lists[stop_pos][:stop_idx]

                return result

            if step == -1 and start > stop:
                result = self.__getitem__(slice(stop + 1, start + 1))
                result.reverse()
                return result

            # Return a list because a negative step could
            # reverse the order of the items and this could
            # be the desired behavior.

            indices = range(start, stop, step)
            return [self.__getitem__(index) for index in indices]
        if self._len:
            if index == 0:
                return lists[0][0]
            if index == -1:
                return lists[-1][-1]
        else:
            msg = "list index out of range"
            raise IndexError(msg)

        if 0 <= index < len(lists[0]):
            return lists[0][index]

        len_last = len(lists[-1])

        if -len_last < index < 0:
            return lists[-1][len_last + index]

        pos, idx = self._pos(index)
        return lists[pos][idx]

    @overload
    def __setitem__(self, index: int, value: T) -> None: ...
    @overload
    def __setitem__(self, index: slice, value: Iterable[T]) -> None: ...
    @override
    def __setitem__(self, index: int | slice, value: T | Iterable[T]) -> None:
        """Raise not-implemented error.

        ``sl.__setitem__(index, value)`` <==> ``sl[index] = value``

        :raises NotImplementedError: use ``del sl[index]`` and
            ``sl.add(value)`` instead

        """
        message = "use ``del sl[index]`` and ``sl.add(value)`` instead"
        raise NotImplementedError(message)

    @override
    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the sorted list.

        ``sl.__iter__()`` <==> ``iter(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return chain.from_iterable(self._lists)

    @override
    def __reversed__(self) -> Iterator[T]:
        """Return a reverse iterator over the sorted list.

        ``sl.__reversed__()`` <==> ``reversed(sl)``

        Iterating the sorted list while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return chain.from_iterable(map(reversed, reversed(self._lists)))

    @override
    def reverse(self):
        """Raise not-implemented error.

        Sorted list maintains values in ascending sort order. Values may not be
        reversed in-place.

        Use ``reversed(sl)`` for an iterator over values in descending sort
        order.

        Implemented to override `MutableSequence.reverse` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``reversed(sl)`` instead

        """
        msg = "use ``reversed(sl)`` instead"
        raise NotImplementedError(msg)

    def islice(
        self, start: int | None = None, stop: int | None = None, reverse: bool = False
    ) -> Iterator[T]:
        """Return an iterator that slices sorted list from `start` to `stop`.

        The `start` and `stop` index are treated inclusive and exclusive,
        respectively.

        Both `start` and `stop` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> sl = SortedList("abcdefghij")
        >>> it = sl.islice(2, 6)
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param int start: start index (inclusive)
        :param int stop: stop index (exclusive)
        :param bool reverse: yield values in reverse order
        :return: iterator

        """
        len_ = self._len

        if not len_:
            return iter(())

        start, stop, _ = slice(start, stop).indices(self._len)

        if start >= stop:
            return iter(())

        pos = self._pos

        min_pos, min_idx = pos(start)

        if stop == len_:
            max_pos = len(self._lists) - 1
            max_idx = len(self._lists[-1])
        else:
            max_pos, max_idx = pos(stop)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    def _islice(
        self, min_pos: int, min_idx: int, max_pos: int, max_idx: int, reverse: bool
    ) -> Iterator[T]:
        """Return an iterator that slices sorted list using two index pairs.

        The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the
        first inclusive and the latter exclusive. See `_pos` for details on how
        an index is converted to an index pair.

        When `reverse` is `True`, values are yielded from the iterator in
        reverse order.

        """
        lists = self._lists

        if min_pos > max_pos:
            return iter(())

        if min_pos == max_pos:
            if reverse:
                indices = reversed(range(min_idx, max_idx))
                return map(lists[min_pos].__getitem__, indices)

            indices = range(min_idx, max_idx)
            return map(lists[min_pos].__getitem__, indices)

        next_pos = min_pos + 1

        if next_pos == max_pos:
            if reverse:
                min_indices = range(min_idx, len(lists[min_pos]))
                max_indices = range(max_idx)
                return chain(
                    map(lists[max_pos].__getitem__, reversed(max_indices)),
                    map(lists[min_pos].__getitem__, reversed(min_indices)),
                )

            min_indices = range(min_idx, len(lists[min_pos]))
            max_indices = range(max_idx)
            return chain(
                map(lists[min_pos].__getitem__, min_indices),
                map(lists[max_pos].__getitem__, max_indices),
            )

        if reverse:
            min_indices = range(min_idx, len(lists[min_pos]))
            sublist_indices = range(next_pos, max_pos)
            sublists = map(lists.__getitem__, reversed(sublist_indices))
            max_indices = range(max_idx)
            return chain(
                map(lists[max_pos].__getitem__, reversed(max_indices)),
                chain.from_iterable(map(reversed, sublists)),
                map(lists[min_pos].__getitem__, reversed(min_indices)),
            )

        min_indices = range(min_idx, len(lists[min_pos]))
        sublist_indices = range(next_pos, max_pos)
        sublists = map(lists.__getitem__, sublist_indices)
        max_indices = range(max_idx)
        return chain(
            map(lists[min_pos].__getitem__, min_indices),
            chain.from_iterable(sublists),
            map(lists[max_pos].__getitem__, max_indices),
        )

    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        reverse: bool = False,
    ) -> Iterator[T]:
        """Create an iterator of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically
        inclusive of the beginning and end of the sorted list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> sl = SortedList("abcdefghij")
        >>> it = sl.irange("c", "f")
        >>> list(it)
        ['c', 'd', 'e', 'f']

        :param minimum: minimum value to start iterating
        :param maximum: maximum value to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order

        Returns:
            `Iterator`

        """
        maxes = self._maxes

        if not maxes:
            return iter(())

        lists = self._lists

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if minimum is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(maxes, minimum)

            if min_pos == len(maxes):
                return iter(())

            min_idx = bisect_left(lists[min_pos], minimum)
        else:
            min_pos = bisect_right(maxes, minimum)

            if min_pos == len(maxes):
                return iter(())

            min_idx = bisect_right(lists[min_pos], minimum)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if maximum is None:
            max_pos = len(maxes) - 1
            max_idx = len(lists[max_pos])
        elif inclusive[1]:
            max_pos = bisect_right(maxes, maximum)

            if max_pos == len(maxes):
                max_pos -= 1
                max_idx = len(lists[max_pos])
            else:
                max_idx = bisect_right(lists[max_pos], maximum)
        else:
            max_pos = bisect_left(maxes, maximum)

            if max_pos == len(maxes):
                max_pos -= 1
                max_idx = len(lists[max_pos])
            else:
                max_idx = bisect_left(lists[max_pos], maximum)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    @override
    def __len__(self) -> int:
        """Return the size of the sorted list.

        ``sl.__len__()`` <==> ``len(sl)``

        :return: size of sorted list

        """
        return self._len

    def bisect_left(self, value: T) -> int:
        """Return an index to insert `value` in the sorted list.

        If the `value` is already present, the insertion point will be before
        (to the left of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([10, 11, 12, 13, 14])
        >>> sl.bisect_left(12)
        2

        :param value: insertion index of value in sorted list
        :return: index

        """
        maxes = self._maxes

        if not maxes:
            return 0

        pos = bisect_left(maxes, value)

        if pos == len(maxes):
            return self._len

        idx = bisect_left(self._lists[pos], value)
        return self._loc(pos, idx)

    def bisect_right(self, value: T) -> int:
        """Return an index to insert `value` in the sorted list.

        Similar to `bisect_left`, but if `value` is already present, the
        insertion point will be after (to the right of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([10, 11, 12, 13, 14])
        >>> sl.bisect_right(12)
        3

        :param value: insertion index of value in sorted list
        :return: index

        """
        maxes = self._maxes

        if not maxes:
            return 0

        pos = bisect_right(maxes, value)

        if pos == len(maxes):
            return self._len

        idx = bisect_right(self._lists[pos], value)
        return self._loc(pos, idx)

    @override
    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
        >>> sl.count(3)
        3

        :param value: value to count in sorted list
        :return: count

        """
        maxes = self._maxes

        if not maxes:
            return 0

        pos_left = bisect_left(maxes, value)

        if pos_left == len(maxes):
            return 0

        lists = self._lists
        idx_left = bisect_left(lists[pos_left], value)
        pos_right = bisect_right(maxes, value)

        if pos_right == len(maxes):
            return self._len - self._loc(pos_left, idx_left)

        idx_right = bisect_right(lists[pos_right], value)

        if pos_left == pos_right:
            return idx_right - idx_left

        right = self._loc(pos_right, idx_right)
        left = self._loc(pos_left, idx_left)
        return right - left

    def copy(self) -> Self:
        """Return a shallow copy of the sorted list.

        Runtime complexity: `O(n)`

        :return: new sorted list

        """
        return self.__class__(self)

    def __copy__(self) -> Self:
        return self.copy()

    @override
    def append(self, value):
        """Raise not-implemented error.

        Implemented to override `MutableSequence.append` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def extend(self, values):
        """Raise not-implemented error.

        Implemented to override `MutableSequence.extend` which provides an
        erroneous default implementation.

        :raises NotImplementedError: use ``sl.update(values)`` instead

        """
        msg = "use ``sl.update(values)`` instead"
        raise NotImplementedError(msg)

    @override
    def insert(self, index: int, value: T) -> None:
        """Raise not-implemented error.

        :raises NotImplementedError: use ``sl.add(value)`` instead

        """
        msg = "use ``sl.add(value)`` instead"
        raise NotImplementedError(msg)

    @override
    def pop(self, index: int = -1) -> T:
        """Remove and return value at `index` in sorted list.

        Raise :exc:`IndexError` if the sorted list is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList("abcde")
        >>> sl.pop()
        'e'
        >>> sl.pop(2)
        'c'
        >>> sl
        SortedList(['a', 'b', 'd'])

        :param int index: index of value (default -1)

        Raises:
            IndexError: if index is out of range

        Returns:
            value

        """
        if not self._len:
            msg = "pop index out of range"
            raise IndexError(msg)

        lists = self._lists

        if index == 0:
            val = lists[0][0]
            self._delete(0, 0)
            return val

        if index == -1:
            pos = len(lists) - 1
            loc = len(lists[pos]) - 1
            val = lists[pos][loc]
            self._delete(pos, loc)
            return val

        if 0 <= index < len(lists[0]):
            val = lists[0][index]
            self._delete(0, index)
            return val

        len_last = len(lists[-1])

        if -len_last < index < 0:
            pos = len(lists) - 1
            loc = len_last + index
            val = lists[pos][loc]
            self._delete(pos, loc)
            return val

        pos, idx = self._pos(index)
        val = lists[pos][idx]
        self._delete(pos, idx)
        return val

    @override
    def index(self, value: T, start: int | None = None, stop: int | None = None) -> int:  # noqa: C901
        """Return first index of value in sorted list.

        Raise ValueError if `value` is not present.

        Index must be between `start` and `stop` for the `value` to be
        considered present. The default value, None, for `start` and `stop`
        indicate the beginning and end of the sorted list.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> sl = SortedList("abcde")
        >>> sl.index("d")
        3
        >>> sl.index("z")
        Traceback (most recent call last):
          ...
        ValueError: 'z' is not in list

        :param value: value in sorted list
        :param int start: start index (default None, start of sorted list)
        :param int stop: stop index (default None, end of sorted list)

        Raises:
            ValueError: if value is not present
        Returns:
            index of value

        """
        len_ = self._len

        if not len_:
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        if start is None:
            start = 0
        if start < 0:
            start += len_
        start = max(start, 0)

        if stop is None:
            stop = len_
        if stop < 0:
            stop += len_
        stop = min(stop, len_)

        if stop <= start:
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        maxes = self._maxes
        pos_left = bisect_left(maxes, value)

        if pos_left == len(maxes):
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        lists = self._lists
        idx_left = bisect_left(lists[pos_left], value)

        if lists[pos_left][idx_left] != value:
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        stop -= 1
        left = self._loc(pos_left, idx_left)

        if start <= left:
            if left <= stop:
                return left
        else:
            right = self.bisect_right(value) - 1

            if start <= right:
                return start

        msg = f"{value!r} is not in list"
        raise ValueError(msg)

    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted list containing all values in both sequences.

        ``sl.__add__(other)`` <==> ``sl + other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(n*log(n))`

        >>> sl1 = SortedList("bat")
        >>> sl2 = SortedList("cat")
        >>> sl1 + sl2
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        :param other: other iterable
        :return: new sorted list

        """
        values: list[T] = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values)

    def __radd__(self, other: Iterable[T]) -> Self:
        return self.__add__(other)

    @override
    def __iadd__(self, other: Iterable[T]) -> Self:
        """Update sorted list with values from `other`.

        ``sl.__iadd__(other)`` <==> ``sl += other``

        Values in `other` do not need to be in sorted order.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> sl = SortedList("bat")
        >>> sl += "cat"
        >>> sl
        SortedList(['a', 'a', 'b', 'c', 't', 't'])

        :param other: other iterable

        Returns:
            existing sorted list

        """
        self.update(other)
        return self

    def __mul__(self, num: int) -> Self:
        """Return new sorted list with `num` shallow copies of values.

        ``sl.__mul__(num)`` <==> ``sl * num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList("abc")
        >>> sl * 3
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        :param int num: count of shallow copies
        :return: new sorted list

        """
        values: list[T] = reduce(iadd, self._lists, []) * num
        return self.__class__(values)

    def __rmul__(self, num: int) -> Self:
        return self.__mul__(num)

    def __imul__(self, num: int) -> Self:
        """Update the sorted list with `num` shallow copies of values.

        ``sl.__imul__(num)`` <==> ``sl *= num``

        Runtime complexity: `O(n*log(n))`

        >>> sl = SortedList("abc")
        >>> sl *= 3
        >>> sl
        SortedList(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])

        :param int num: count of shallow copies

        Returns:
            existing sorted list

        """
        values: list[T] = reduce(iadd, self._lists, []) * num
        self.clear()
        self.update(values)
        return self

    @staticmethod
    def __make_cmp(seq_op: Callable[[object, object], bool], symbol: str, doc: str):
        """Make comparator method."""

        def comparer(self: Self, other: object) -> NotImplementedType | bool:
            """Compare method for sorted list and sequence."""
            if not isinstance(other, Sequence):
                return NotImplemented

            self_len = self._len
            len_other = len(other)

            if self_len != len_other:
                if seq_op is eq:
                    return False
                if seq_op is ne:
                    return True

            for alpha, beta in zip(self, other, strict=False):
                if alpha != beta:
                    return seq_op(alpha, beta)

            return seq_op(self_len, len_other)

        seq_op_name = seq_op.__name__
        comparer.__name__ = f"__{seq_op_name}__"
        doc_str = """Return true if and only if sorted list is {0} `other`.

        ``sl.__{1}__(other)`` <==> ``sl {2} other``

        Comparisons use lexicographical order as with sequences.

        Runtime complexity: `O(n)`

        :param other: `other` sequence
        :return: true if sorted list is {0} `other`

        """
        comparer.__doc__ = dedent(doc_str.format(doc, seq_op_name, symbol))
        return comparer

    __eq__ = __make_cmp(eq, "==", "equal to")
    __ne__ = __make_cmp(ne, "!=", "not equal to")
    __lt__ = __make_cmp(lt, "<", "less than")
    __gt__ = __make_cmp(gt, ">", "greater than")
    __le__ = __make_cmp(le, "<=", "less than or equal to")
    __ge__ = __make_cmp(ge, ">=", "greater than or equal to")

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[list[T]]]:
        values: list[T] = reduce(iadd, self._lists, [])
        return (type(self), (values,))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted list.

        ``sl.__repr__()`` <==> ``repr(sl)``

        :return: string representation

        """
        return f"{type(self).__name__}({list(self)!r})"


def identity[T](value: T) -> T:
    """Identity function."""
    return value


class SortedKeyList[T, OT: SupportsRichComparison](SortedList[T]):  # pyright: ignore[reportInvalidTypeArguments]
    """Sorted-key list is a subtype of sorted list.

    The sorted-key list maintains values in comparison order based on the
    result of a key function applied to every value.

    All the same methods that are available in :class:`SortedList` are also
    available in :class:`SortedKeyList`.

    Additional methods provided:

    * :attr:`SortedKeyList.key`
    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Some examples below use:

    >>> from operator import neg
    >>> neg
    <built-in function neg>
    >>> neg(1)
    -1

    """

    def __init__(
        self, iterable: Iterable[T] | None = None, key: KeyFunc[T, OT] = identity
    ) -> None:
        """Initialize sorted-key list instance.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted-key list.

        Optional `key` argument defines a callable that, like the `key`
        argument to Python's `sorted` function, extracts a comparison key from
        each value. The default is the identity function.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl
        SortedKeyList([], key=<built-in function neg>)
        >>> skl = SortedKeyList([3, 1, 2], key=neg)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param iterable: initial values (optional)
        :param key: function used to extract comparison key (optional)

        """
        self._key: KeyFunc[T, OT] = key
        self._len: int = 0
        self._load: int = self.DEFAULT_LOAD_FACTOR
        self._lists: list[list[T]] = []
        self._keys: list[list[OT]] = []
        self._maxes: list[OT] = []
        self._index: list[int] = []
        self._offset: int = 0

        if iterable is not None:
            self.update(iterable)

    @property
    def key(self) -> KeyFunc[T, OT]:
        """Function used to extract comparison key from values."""
        return self._key

    @override
    def clear(self) -> None:
        """Remove all values from sorted-key list.

        Runtime complexity: `O(n)`

        """
        self._len = 0
        del self._lists[:]
        del self._keys[:]
        del self._maxes[:]
        del self._index[:]

    @override
    def add(self, value: T) -> None:
        """Add `value` to sorted-key list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.add(3)
        >>> skl.add(1)
        >>> skl.add(2)
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param value: value to add to sorted-key list

        """
        lists = self._lists
        keys = self._keys
        maxes = self._maxes

        key = self._key(value)

        if maxes:
            pos = bisect_right(maxes, key)

            if pos == len(maxes):
                pos -= 1
                lists[pos].append(value)
                keys[pos].append(key)
                maxes[pos] = key
            else:
                idx = bisect_right(keys[pos], key)
                lists[pos].insert(idx, value)
                keys[pos].insert(idx, key)

            self._expand(pos)
        else:
            lists.append([value])
            keys.append([key])
            maxes.append(key)

        self._len += 1

    @override
    def _expand(self, pos: int) -> None:
        """Split sublists with length greater than double the load-factor.

        Updates the index when the sublist length is less than double the load
        level. This requires incrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        """
        lists = self._lists
        keys = self._keys
        index = self._index

        if len(keys[pos]) > (self._load << 1):
            maxes = self._maxes
            load = self._load

            lists_pos = lists[pos]
            keys_pos = keys[pos]
            half = lists_pos[load:]
            half_keys = keys_pos[load:]
            del lists_pos[load:]
            del keys_pos[load:]
            maxes[pos] = keys_pos[-1]

            lists.insert(pos + 1, half)
            keys.insert(pos + 1, half_keys)
            maxes.insert(pos + 1, half_keys[-1])

            del index[:]
        elif index:
            child = self._offset + pos
            while child:
                index[child] += 1
                child = (child - 1) >> 1
            index[0] += 1

    @override
    def update(self, iterable: Iterable[T]) -> None:
        """Update sorted-key list by adding all values from `iterable`.

        Runtime complexity: `O(k*log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList(key=neg)
        >>> skl.update([3, 1, 2])
        >>> skl
        SortedKeyList([3, 2, 1], key=<built-in function neg>)

        :param iterable: iterable of values to add

        """
        lists = self._lists
        keys = self._keys
        maxes = self._maxes
        values = sorted(iterable, key=self._key)

        if maxes:
            if len(values) * 4 >= self._len:
                lists.append(values)
                values: list[T] = reduce(iadd, lists, [])
                values.sort(key=self._key)
                self.clear()
            else:
                add_ = self.add
                for val in values:
                    add_(val)
                return

        load = self._load
        lists.extend(values[pos : (pos + load)] for pos in range(0, len(values), load))
        keys.extend(list(map(self._key, list_)) for list_ in lists)
        maxes.extend(sublist[-1] for sublist in keys)
        self._len = len(values)
        del self._index[:]

    @override
    def __contains__(self, value: object) -> bool:
        """Return true if `value` is an element of the sorted-key list.

        ``skl.__contains__(value)`` <==> ``value in skl``

        Runtime complexity: `O(log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> 3 in skl
        True

        :param value: search for value in sorted-key list
        :return: true if `value` in sorted-key list

        """
        maxes = self._maxes

        if not maxes:
            return False

        key = self._key(value)
        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            return False

        lists = self._lists
        keys = self._keys

        idx = bisect_left(keys[pos], key)

        len_keys = len(keys)
        len_sublist = len(keys[pos])

        while True:
            if keys[pos][idx] != key:
                return False
            if lists[pos][idx] == value:
                return True
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return False
                len_sublist = len(keys[pos])
                idx = 0

    @override
    def discard(self, value: T) -> None:
        """Remove `value` from sorted-key list if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.discard(1)
        >>> skl.discard(0)
        >>> skl == [5, 4, 3, 2]
        True

        :param value: `value` to discard from sorted-key list

        """
        maxes = self._maxes

        if not maxes:
            return

        key = self._key(value)
        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            return

        lists = self._lists
        keys = self._keys
        idx = bisect_left(keys[pos], key)
        len_keys = len(keys)
        len_sublist = len(keys[pos])

        while True:
            if keys[pos][idx] != key:
                return
            if lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return
                len_sublist = len(keys[pos])
                idx = 0

    @override
    def remove(self, value: T) -> None:
        """Remove `value` from sorted-key list; `value` must be a member.

        If `value` is not a member, raise ValueError.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([1, 2, 3, 4, 5], key=neg)
        >>> skl.remove(5)
        >>> skl == [4, 3, 2, 1]
        True
        >>> skl.remove(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 not in list

        :param value: `value` to remove from sorted-key list

        Raises:
            ValueError: if `value` is not in sorted-key list

        """
        maxes = self._maxes

        if not maxes:
            msg = f"{value!r} not in list"
            raise ValueError(msg)

        key = self._key(value)
        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            msg = f"{value!r} not in list"
            raise ValueError(msg)

        lists = self._lists
        keys = self._keys
        idx = bisect_left(keys[pos], key)
        len_keys = len(keys)
        len_sublist = len(keys[pos])

        while True:
            if keys[pos][idx] != key:
                msg_0 = f"{value!r} not in list"
                raise ValueError(msg_0)
            if lists[pos][idx] == value:
                self._delete(pos, idx)
                return
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    msg = f"{value!r} not in list"
                    raise ValueError(msg)
                len_sublist = len(keys[pos])
                idx = 0

    @override
    def _delete(self, pos: int, idx: int) -> None:
        """Delete value at the given `(pos, idx)`.

        Combines lists that are less than half the load level.

        Updates the index when the sublist length is more than half the load
        level. This requires decrementing the nodes in a traversal from the
        leaf node to the root. For an example traversal see
        ``SortedList._loc``.

        :param int pos: lists index
        :param int idx: sublist index

        """
        lists = self._lists
        keys = self._keys
        maxes = self._maxes
        index = self._index
        keys_pos = keys[pos]
        lists_pos = lists[pos]

        del keys_pos[idx]
        del lists_pos[idx]
        self._len -= 1

        len_keys_pos = len(keys_pos)

        if len_keys_pos > (self._load >> 1):
            maxes[pos] = keys_pos[-1]

            if index:
                child = self._offset + pos
                while child > 0:
                    index[child] -= 1
                    child = (child - 1) >> 1
                index[0] -= 1
        elif len(keys) > 1:
            if not pos:
                pos += 1

            prev = pos - 1
            keys[prev].extend(keys[pos])
            lists[prev].extend(lists[pos])
            maxes[prev] = keys[prev][-1]

            del lists[pos]
            del keys[pos]
            del maxes[pos]
            del index[:]

            self._expand(prev)
        elif len_keys_pos:
            maxes[pos] = keys_pos[-1]
        else:
            del lists[pos]
            del keys[pos]
            del maxes[pos]
            del index[:]

    @override
    def irange(
        self,
        minimum: T | None = None,
        maximum: T | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        reverse: bool = False,
    ) -> Iterator[T]:
        """Create an iterator of values between `minimum` and `maximum`.

        Both `minimum` and `maximum` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange(14.5, 11.5)
        >>> list(it)
        [14, 13, 12]

        :param minimum: minimum value to start iterating
        :param maximum: maximum value to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order

        Returns:
            `Iterator`

        """
        min_key = self._key(minimum) if minimum is not None else None
        max_key = self._key(maximum) if maximum is not None else None
        return self.irange_key(
            min_key=min_key,
            max_key=max_key,
            inclusive=inclusive,
            reverse=reverse,
        )

    def irange_key(  # noqa: PLR0912
        self,
        min_key: OT | None = None,
        max_key: OT | None = None,
        inclusive: tuple[bool, bool] = (True, True),
        reverse: bool = False,
    ) -> Iterator[T]:
        """Create an iterator of values between `min_key` and `max_key`.

        Both `min_key` and `max_key` default to `None` which is automatically
        inclusive of the beginning and end of the sorted-key list.

        The argument `inclusive` is a pair of booleans that indicates whether
        the minimum and maximum ought to be included in the range,
        respectively. The default is ``(True, True)`` such that the range is
        inclusive of both minimum and maximum.

        When `reverse` is `True` the values are yielded from the iterator in
        reverse order; `reverse` defaults to `False`.

        >>> from operator import neg
        >>> skl = SortedKeyList([11, 12, 13, 14, 15], key=neg)
        >>> it = skl.irange_key(-14, -12)
        >>> list(it)
        [14, 13, 12]

        :param min_key: minimum key to start iterating
        :param max_key: maximum key to stop iterating
        :param inclusive: pair of booleans
        :param bool reverse: yield values in reverse order

        Returns:
            `Iterator`

        """
        maxes = self._maxes

        if not maxes:
            return iter(())

        keys = self._keys

        # Calculate the minimum (pos, idx) pair. By default this location
        # will be inclusive in our calculation.

        if min_key is None:
            min_pos = 0
            min_idx = 0
        elif inclusive[0]:
            min_pos = bisect_left(maxes, min_key)

            if min_pos == len(maxes):
                return iter(())

            min_idx = bisect_left(keys[min_pos], min_key)
        else:
            min_pos = bisect_right(maxes, min_key)

            if min_pos == len(maxes):
                return iter(())

            min_idx = bisect_right(keys[min_pos], min_key)

        # Calculate the maximum (pos, idx) pair. By default this location
        # will be exclusive in our calculation.

        if max_key is None:
            max_pos = len(maxes) - 1
            max_idx = len(keys[max_pos])
        elif inclusive[1]:
            max_pos = bisect_right(maxes, max_key)

            if max_pos == len(maxes):
                max_pos -= 1
                max_idx = len(keys[max_pos])
            else:
                max_idx = bisect_right(keys[max_pos], max_key)
        else:
            max_pos = bisect_left(maxes, max_key)

            if max_pos == len(maxes):
                max_pos -= 1
                max_idx = len(keys[max_pos])
            else:
                max_idx = bisect_left(keys[max_pos], max_key)

        return self._islice(min_pos, min_idx, max_pos, max_idx, reverse)

    @override
    def bisect_left(self, value: T) -> int:
        """Return an index to insert `value` in the sorted-key list.

        If the `value` is already present, the insertion point will be before
        (to the left of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_left(1)
        4

        :param value: insertion index of value in sorted-key list
        :return: index

        """
        return self.bisect_key_left(self._key(value))

    @override
    def bisect_right(self, value: T) -> int:
        """Return an index to insert `value` in the sorted-key list.

        Similar to `bisect_left`, but if `value` is already present, the
        insertion point will be after (to the right of) any existing values.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_right(1)
        5

        :param value: insertion index of value in sorted-key list
        :return: index

        """
        return self.bisect_key_right(self._key(value))

    def bisect_key_left(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        If the `key` is already present, the insertion point will be before (to
        the left of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_left(-1)
        4

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        maxes = self._maxes

        if not maxes:
            return 0

        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            return self._len

        idx = bisect_left(self._keys[pos], key)

        return self._loc(pos, idx)

    def bisect_key_right(self, key: OT) -> int:
        """Return an index to insert `key` in the sorted-key list.

        Similar to `bisect_key_left`, but if `key` is already present, the
        insertion point will be after (to the right of) any existing keys.

        Similar to the `bisect` module in the standard library.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.bisect_key_right(-1)
        5

        :param key: insertion index of key in sorted-key list
        :return: index

        """
        maxes = self._maxes

        if not maxes:
            return 0

        pos = bisect_right(maxes, key)

        if pos == len(maxes):
            return self._len

        idx = bisect_right(self._keys[pos], key)

        return self._loc(pos, idx)

    @override
    def count(self, value: T) -> int:
        """Return number of occurrences of `value` in the sorted-key list.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([4, 4, 4, 4, 3, 3, 3, 2, 2, 1], key=neg)
        >>> skl.count(2)
        2

        :param value: value to count in sorted-key list
        :return: count

        """
        maxes = self._maxes

        if not maxes:
            return 0

        key = self._key(value)
        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            return 0

        lists = self._lists
        keys = self._keys
        idx = bisect_left(keys[pos], key)
        total = 0
        len_keys = len(keys)
        len_sublist = len(keys[pos])

        while True:
            if keys[pos][idx] != key:
                return total
            if lists[pos][idx] == value:
                total += 1
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    return total
                len_sublist = len(keys[pos])
                idx = 0

    @override
    def copy(self) -> Self:
        """Return a shallow copy of the sorted-key list.

        Runtime complexity: `O(n)`

        :return: new sorted-key list

        """
        return self.__class__(self, key=self._key)

    @override
    def index(self, value: T, start: int | None = None, stop: int | None = None) -> int:  # noqa: C901, PLR0912
        """Return first index of value in sorted-key list.

        Raise ValueError if `value` is not present.

        Index must be between `start` and `stop` for the `value` to be
        considered present. The default value, None, for `start` and `stop`
        indicate the beginning and end of the sorted-key list.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> from operator import neg
        >>> skl = SortedKeyList([5, 4, 3, 2, 1], key=neg)
        >>> skl.index(2)
        3
        >>> skl.index(0)
        Traceback (most recent call last):
          ...
        ValueError: 0 is not in list

        :param value: value in sorted-key list
        :param int start: start index (default None, start of sorted-key list)
        :param int stop: stop index (default None, end of sorted-key list)

        Raises:
            ValueError: if value is not present
        Returns:
            index of value

        """
        len_ = self._len

        if not len_:
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        if start is None:
            start = 0
        if start < 0:
            start += len_
        start = max(start, 0)

        if stop is None:
            stop = len_
        if stop < 0:
            stop += len_
        stop = min(stop, len_)

        if stop <= start:
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        maxes = self._maxes
        key = self._key(value)
        pos = bisect_left(maxes, key)

        if pos == len(maxes):
            msg = f"{value!r} is not in list"
            raise ValueError(msg)

        stop -= 1
        lists = self._lists
        keys = self._keys
        idx = bisect_left(keys[pos], key)
        len_keys = len(keys)
        len_sublist = len(keys[pos])

        while True:
            if keys[pos][idx] != key:
                msg_0 = f"{value!r} is not in list"
                raise ValueError(msg_0)
            if lists[pos][idx] == value:
                loc = self._loc(pos, idx)
                if start <= loc <= stop:
                    return loc
                if loc > stop:
                    break
            idx += 1
            if idx == len_sublist:
                pos += 1
                if pos == len_keys:
                    msg = f"{value!r} is not in list"
                    raise ValueError(msg)
                len_sublist = len(keys[pos])
                idx = 0

        msg = f"{value!r} is not in list"
        raise ValueError(msg)

    @override
    def __add__(self, other: Iterable[T]) -> Self:
        """Return new sorted-key list containing all values in both sequences.

        ``skl.__add__(other)`` <==> ``skl + other``

        Values in `other` do not need to be in sorted-key order.

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl1 = SortedKeyList([5, 4, 3], key=neg)
        >>> skl2 = SortedKeyList([2, 1, 0], key=neg)
        >>> skl1 + skl2
        SortedKeyList([5, 4, 3, 2, 1, 0], key=<built-in function neg>)

        :param other: other iterable
        :return: new sorted-key list

        """
        values: list[T] = reduce(iadd, self._lists, [])
        values.extend(other)
        return self.__class__(values, key=self._key)

    @override
    def __mul__(self, num: int) -> Self:
        """Return new sorted-key list with `num` shallow copies of values.

        ``skl.__mul__(num)`` <==> ``skl * num``

        Runtime complexity: `O(n*log(n))`

        >>> from operator import neg
        >>> skl = SortedKeyList([3, 2, 1], key=neg)
        >>> skl * 2
        SortedKeyList([3, 3, 2, 2, 1, 1], key=<built-in function neg>)

        :param int num: count of shallow copies
        :return: new sorted-key list

        """
        values: list[T] = reduce(iadd, self._lists, []) * num
        return self.__class__(values, key=self._key)

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[list[T], KeyFunc[T, OT]]]:
        values: list[T] = reduce(iadd, self._lists, [])
        return (type(self), (values, self.key))

    @recursive_repr()
    def __repr__(self) -> str:
        """Return string representation of sorted-key list.

        ``skl.__repr__()`` <==> ``repr(skl)``

        :return: string representation

        """
        type_name = type(self).__name__
        return f"{type_name}({list(self)!r}, key={self._key!r})"
