from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import TYPE_CHECKING, Any, Concatenate

import cytoolz as cz

from .._core import MappingWrapper

if TYPE_CHECKING:
    from .._iter import Iter, Seq
    from ._main import Dict


class IterDict[K, V](MappingWrapper[K, V]):
    def itr[**P, R, U](
        self: MappingWrapper[K, Iterable[U]],
        func: Callable[Concatenate[Iter[U], P], R],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> Dict[K, R]:
        """
        Apply a function to each value after wrapping it in an Iter.

        Args:
            func: Function to apply to each value after wrapping it in an Iter.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Syntactic sugar for `map_values(lambda data: func(Iter(data), *args, **kwargs))`
        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "numbers1": [1, 2, 3],
        ...     "numbers2": [4, 5, 6],
        ... }
        >>> pc.Dict(data).itr(lambda v: v.repeat(5).flatten().sum()).inner()
        {'numbers1': 30, 'numbers2': 75}

        ```
        """
        from .._iter import Iter

        def _itr(data: Mapping[K, Iterable[U]]) -> dict[K, R]:
            def _(v: Iterable[U]) -> R:
                return func(Iter(iter(v)), *args, **kwargs)

            return cz.dicttoolz.valmap(_, data)

        return self._new(_itr)

    def iter_keys(self) -> Iter[K]:
        """
        Return an Iter of the dict's keys.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).iter_keys().into(list)
        [1]

        ```
        """
        from .._iter import Iter

        def _keys(data: dict[K, V]) -> Iter[K]:
            return Iter(iter(data.keys()))

        return self.into(_keys)

    def iter_values(self) -> Iter[V]:
        """
        Return an Iter of the dict's values.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).iter_values().into(list)
        [2]

        ```
        """
        from .._iter import Iter

        def _values(data: dict[K, V]) -> Iter[V]:
            return Iter(iter(data.values()))

        return self.into(_values)

    def iter_items(self) -> Iter[tuple[K, V]]:
        """
        Return an Iter of the dict's items.
        ```python
        >>> import pyochain as pc
        >>> pc.Dict({1: 2}).iter_items().into(list)
        [(1, 2)]

        ```
        """
        from .._iter import Iter

        def _items(data: dict[K, V]) -> Iter[tuple[K, V]]:
            return Iter(iter(data.items()))

        return self.into(_items)

    def to_arrays(self) -> Seq[list[Any]]:
        """
        Convert the nested dictionary into a sequence of arrays.

        The sequence represents all paths from root to leaves.

        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "a": {"b": 1, "c": 2},
        ...     "d": {"e": {"f": 3}},
        ... }
        >>> pc.Dict(data).to_arrays().inner()
        [['a', 'b', 1], ['a', 'c', 2], ['d', 'e', 'f', 3]]

        ```
        """
        from .._iter import Seq

        def _to_arrays(d: Mapping[Any, Any]) -> list[list[Any]]:
            """from dictutils.pivot"""
            match d:
                case Mapping():
                    arr: list[Any] = []
                    for k, v in d.items():
                        for el in _to_arrays(v):
                            arr.append([k] + el)
                    return arr

                case _:
                    return [[d]]

        return Seq(self.into(_to_arrays))
