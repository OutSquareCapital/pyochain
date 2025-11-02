from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Concatenate

from .._core import SupportsKeysAndGetItem, dict_repr
from ._exprs import IntoExpr, compute_exprs
from ._filters import FilterDict
from ._groups import GroupsDict
from ._iter import IterDict
from ._joins import JoinsDict
from ._maps import MapDict
from ._nested import NestedDict
from ._process import ProcessDict


class DictCommonMethods[K, V](
    ProcessDict[K, V],
    IterDict[K, V],
    NestedDict[K, V],
    MapDict[K, V],
    JoinsDict[K, V],
    FilterDict[K, V],
    GroupsDict[K, V],
):
    def __repr__(self) -> str:
        return f"{self.into(dict_repr)}"

    def select(
        self: DictCommonMethods[str, Any], *exprs: IntoExpr
    ) -> LazyDict[str, Any]:
        """
        Select and alias fields from the dict based on expressions and/or strings.

        Navigate nested fields using the `pyochain.key` function.

        - Chain `key.key()` calls to access nested fields.
        - Use `key.apply()` to transform values.
        - Use `key.alias()` to rename fields in the resulting dict.

        Args:
            *exprs: Expressions or strings to select and alias fields from the dictionary.

        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "name": "Alice",
        ...     "age": 30,
        ...     "scores": {"eng": [85, 90, 95], "math": [80, 88, 92]},
        ... }
        >>> scores_expr = pc.key("scores")  # save an expression for reuse
        >>> pc.Dict(data).select(
        ...     pc.key("name").alias("student_name"),
        ...     "age",  # shorthand for pc.key("age")
        ...     scores_expr.key("math").alias("math_scores"),
        ...     scores_expr.key("eng")
        ...     .apply(lambda v: pc.Seq(v).mean())
        ...     .alias("average_eng_score"),
        ... ).unwrap()
        {'student_name': 'Alice', 'age': 30, 'math_scores': [80, 88, 92], 'average_eng_score': 90}

        ```
        """

        def _select(data: dict[str, Any]) -> dict[str, Any]:
            return compute_exprs(exprs, data, {})

        return self._new(_select)

    def with_fields(
        self: DictCommonMethods[str, Any], *exprs: IntoExpr
    ) -> LazyDict[str, Any]:
        """
        Merge aliased expressions into the root dict (overwrite on collision).

        Args:
            *exprs: Expressions to merge into the root dictionary.

        ```python
        >>> import pyochain as pc
        >>> data = {
        ...     "name": "Alice",
        ...     "age": 30,
        ...     "scores": {"eng": [85, 90, 95], "math": [80, 88, 92]},
        ... }
        >>> scores_expr = pc.key("scores")  # save an expression for reuse
        >>> pc.Dict(data).with_fields(
        ...     scores_expr.key("eng")
        ...     .apply(lambda v: pc.Seq(v).mean())
        ...     .alias("average_eng_score"),
        ... ).unwrap()
        {'name': 'Alice', 'age': 30, 'scores': {'eng': [85, 90, 95], 'math': [80, 88, 92]}, 'average_eng_score': 90}

        ```
        """

        def _with_fields(data: dict[str, Any]) -> dict[str, Any]:
            return compute_exprs(exprs, data, data.copy())

        return self._new(_with_fields)

    def apply[**P, KU, VU](
        self,
        func: Callable[Concatenate[dict[K, V], P], dict[KU, VU]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> LazyDict[KU, VU]:
        """
        Apply a function to the underlying dict and return a Dict of the result.
        Allow to pass user defined functions that transform the dict while retaining the Dict wrapper.

        Args:
            func: Function to apply to the underlying dict.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.
        Example:
        ```python
        >>> import pyochain as pc
        >>> def invert_dict(d: dict[K, V]) -> dict[V, K]:
        ...     return {v: k for k, v in d.items()}
        >>> pc.Dict({'a': 1, 'b': 2}).apply(invert_dict).unwrap()
        {1: 'a', 2: 'b'}

        ```
        """

        def _(data: dict[K, V]) -> dict[KU, VU]:
            return func(data, *args, **kwargs)

        return self._new(_)


class Dict[K, V](DictCommonMethods[K, V]):
    """
    Wrapper for Python dictionaries with chainable methods.
    """

    __slots__ = ()

    def _new[KU, VU](
        self, func: Callable[[dict[K, V]], dict[KU, VU]]
    ) -> LazyDict[KU, VU]:
        def node() -> dict[KU, VU]:
            return func(self.unwrap())

        return LazyDict(node)

    def lazy(self) -> LazyDict[K, V]:
        """
        Convert to a LazyDict for lazy evaluation of operations.

        Returns:
            A LazyDict instance wrapping the current Dict.
        """
        return LazyDict(self.unwrap())

    @staticmethod
    def from_[G, I](
        data: Mapping[G, I] | Iterable[tuple[G, I]] | SupportsKeysAndGetItem[G, I],
    ) -> Dict[G, I]:
        """
        Create a Dict from a convertible value.

        Args:
            data: A mapping, Iterable of tuples, or object supporting keys and item access to convert into a Dict.

        Returns:
            A Dict instance containing the data from the input.

        Example:

        ```python
        >>> import pyochain as pc
        >>> class MyMapping:
        ...     def __init__(self):
        ...         self._data = {1: "a", 2: "b", 3: "c"}
        ...
        ...     def keys(self):
        ...         return self._data.keys()
        ...
        ...     def __getitem__(self, key):
        ...         return self._data[key]
        >>>
        >>> pc.Dict.from_(MyMapping()).unwrap()
        {1: 'a', 2: 'b', 3: 'c'}
        >>> pc.Dict.from_([("d", "e"), ("f", "g")]).unwrap()
        {'d': 'e', 'f': 'g'}

        ```
        """
        return Dict(dict(data))

    @staticmethod
    def from_object(obj: object) -> Dict[str, Any]:
        """
        Create a Dict from an object's __dict__ attribute.

        Args:
            obj: The object whose `__dict__` attribute will be used to create the Dict.

        ```python
        >>> import pyochain as pc
        >>> class Person:
        ...     def __init__(self, name: str, age: int):
        ...         self.name = name
        ...         self.age = age
        >>> person = Person("Alice", 30)
        >>> pc.Dict.from_object(person).unwrap()
        {'name': 'Alice', 'age': 30}

        ```
        """
        return Dict(obj.__dict__)

    def pivot(self, *indices: int) -> Dict[Any, Any]:
        """
        Pivot a nested dictionary by rearranging the key levels according to order.

        Syntactic sugar for to_arrays().rearrange(*indices).to_records()

        Args:
            indices: Indices specifying the new order of key levels

        Returns:
            Pivoted dictionary with keys rearranged

        Example:
        ```python
        >>> import pyochain as pc
        >>> d = {"A": {"X": 1, "Y": 2}, "B": {"X": 3, "Y": 4}}
        >>> pc.Dict(d).pivot(1, 0).unwrap()
        {'X': {'A': 1, 'B': 3}, 'Y': {'A': 2, 'B': 4}}
        """

        return self.to_arrays().rearrange(*indices).to_records()


class LazyDict[K, V](DictCommonMethods[K, V]):
    _node: Callable[[], dict[K, V]]
    __slots__ = ("_node",)

    def __init__(self, data: dict[K, V] | Callable[[], dict[K, V]]) -> None:
        self._node = lambda: data() if callable(data) else data

    def _new[KU, VU](
        self, func: Callable[[dict[K, V]], dict[KU, VU]]
    ) -> LazyDict[KU, VU]:
        def new_chained_node() -> dict[KU, VU]:
            return func(self._node())

        return LazyDict(new_chained_node)

    def collect(self) -> Dict[K, V]:
        """
        Evaluate all lazy operations and return a Dict with the result."""
        return Dict(self._node())

    def unwrap(self) -> dict[K, V]:
        """
        Unwrap and return the underlying dictionary.
        Returns:
            The underlying dictionary.

        Note:
            This forces evaluation of all lazy operations.
        """
        return self.collect().unwrap()
