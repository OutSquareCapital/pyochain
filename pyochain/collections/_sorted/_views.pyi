# Adapted from python-sortedcontainers (https://github.com/grantjenks/python-sortedcontainers)
# Copyright 2014-2024 Grant Jenks — Licensed under the Apache License 2.0

from collections.abc import Iterable
from typing import Any, Generic, TypeVar, overload, override

from pyochain import Vec
from pyochain.abc import PyoItemsView, PyoKeysView, PyoSequence, PyoValuesView
from pyochain.collections import SortedSet

from ..._types import SupportsHashableAndRichComparison

_K_co = TypeVar("_K_co", covariant=True, bound=SupportsHashableAndRichComparison)
_V_co = TypeVar("_V_co", covariant=True)

class BaseSortedView(Generic[_K_co, _V_co]):  # ruff: ignore[non-pep695-generic-class]
    def __delitem__[K: SupportsHashableAndRichComparison, V](
        self, index: int | slice
    ) -> None:
        """Remove item at `index` from sorted dict.

        ``view.__delitem__(index)`` <==> ``del view[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Examples:
            ```python
            from pyochain.collections import SortedDict

            sd = SortedDict({"a": 1, "b": 2, "c": 3})
            view = sd.keys()
            del view[0]
            assert sd == SortedDict({"b": 2, "c": 3})
            del view[-1]
            assert sd == SortedDict({"b": 2})
            del view[:]
            assert sd == SortedDict({})
            with pytest.raises(IndexError):
                del view[10]
            ```
        """

class SortedKeysView(
    BaseSortedView[_K_co, Any],
    PyoKeysView[_K_co],
    PyoSequence[_K_co],
    Generic[_K_co],  # ruff: ignore[non-pep695-generic-class]
):
    """Sorted keys view is a dynamic view of the sorted dict's keys.

    When the sorted dict's keys change, the view reflects those changes.

    The keys view implements the set and sequence abstract base classes.

    """

    @classmethod
    @override
    # pyrefly: ignore [bad-override]
    def _from_iterable(cls, it: Iterable[_K_co]) -> SortedSet[_K_co]:  # pyright: ignore[reportIncompatibleMethodOverride]
        ...
    @overload
    def __getitem__(self, index: int) -> _K_co: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[_K_co]: ...
    @override
    def __getitem__(self, index: int | slice) -> _K_co | Vec[_K_co]:
        """Lookup key at `index` in sorted keys views.

        ``skv.__getitem__(index)`` <==> ``skv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            _K_co | Vec[_K_co]: key or list of keys

        Examples:
            ```python
            from pyochain import Ok, Err, Vec
            from pyochain.collections import SortedDict
            import pytest

            sd = SortedDict({"a": 1, "b": 2, "c": 3})
            skv = sd.keys()
            assert skv[0] == "a"
            assert skv[-1] == "c"
            assert skv[:] == Vec(("a", "b", "c"))
            with pytest.raises(IndexError):
                skv[100]
            ```
        """

class SortedItemsView(
    BaseSortedView[_K_co, _V_co],
    PyoItemsView[_K_co, _V_co],
    PyoSequence[tuple[_K_co, _V_co]],
    Generic[_K_co, _V_co],  # ruff:ignore[non-pep695-generic-class]
):
    """Sorted items view is a dynamic view of the sorted dict's items.

    When the sorted dict's items change, the view reflects those changes.

    The items view implements the set and sequence abstract base classes.

    """

    @classmethod
    @override
    # pyrefly: ignore [bad-override]
    def _from_iterable(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, it: Iterable[tuple[_K_co, _V_co]]
    ) -> SortedSet[tuple[_K_co, _V_co]]: ...
    @overload
    def __getitem__(self, index: int) -> tuple[_K_co, _V_co]: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[tuple[_K_co, _V_co]]: ...
    @override
    def __getitem__(
        self, index: int | slice
    ) -> tuple[_K_co, _V_co] | Vec[tuple[_K_co, _V_co]]:
        """Lookup item at `index` in sorted items view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            tuple[_K_co, _V_co] | Vec[tuple[_K_co, _V_co]]: item or list of items

        Examples:
            ```python
            from pyochain import Vec
            from pyochain.collections import SortedDict
            import pytest

            sd = SortedDict({"a": 1, "b": 2, "c": 3})
            siv = sd.items()
            assert siv[0] == ("a", 1)
            assert siv[-1] == ("c", 3)
            assert siv[:] == Vec([("a", 1), ("b", 2), ("c", 3)])
            with pytest.raises(IndexError):
                siv[100]
            ```

        """

class SortedValuesView(
    BaseSortedView[Any, _V_co],
    PyoValuesView[_V_co],
    PyoSequence[_V_co],
    Generic[_V_co],  # ruff: ignore[non-pep695-generic-class]
):
    """Sorted values view is a dynamic view of the sorted dict's values.

    When the sorted dict's values change, the view reflects those changes.

    The values view implements the sequence abstract base class.

    """

    @overload
    def __getitem__(self, index: int) -> _V_co: ...
    @overload
    def __getitem__(self, index: slice) -> Vec[_V_co]: ...
    @override
    def __getitem__(self, index: int | slice) -> _V_co | Vec[_V_co]:
        """Lookup value at `index` in sorted values view.

        ``siv.__getitem__(index)`` <==> ``siv[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        Args:
            index (int | slice): integer or slice for indexing

        Returns:
            _V_co | Vec[_V_co]: value or list of values

        Examples:
            ```python
            from pyochain import Vec
            from pyochain.collections import SortedDict
            import pytest

            sd = SortedDict({"a": 2, "b": 1, "c": 3})
            svv = sd.values()
            assert svv[0] == 2
            assert svv[-1] == 3
            assert svv[:] == Vec((2, 1, 3))
            with pytest.raises(IndexError):
                svv[100]
            ```
        """
