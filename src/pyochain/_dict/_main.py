from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Self

from .._core import SupportsKeysAndGetItem
from ._exprs import IntoExpr, compute_exprs
from ._filters import FilterDict
from ._groups import GroupsDict
from ._iter import IterDict
from ._joins import JoinsDict
from ._maps import MapDict
from ._nested import NestedDict
from ._process import ProcessDict


class Dict[K, V](
    ProcessDict[K, V],
    IterDict[K, V],
    NestedDict[K, V],
    MapDict[K, V],
    JoinsDict[K, V],
    FilterDict[K, V],
    GroupsDict[K, V],
):
    """
    Wrapper for Python dictionaries with chainable methods.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        def dict_repr(
            v: Mapping[Any, Any] | list[Any] | str,
            depth: int = 0,
            max_depth: int = 3,
            max_items: int = 6,
            max_str: int = 80,
            indent: int = 2,
        ) -> str:
            pad = " " * (depth * indent)
            if depth > max_depth:
                return "…"
            match v:
                case Mapping():
                    items: list[tuple[str, Any]] = list(v.items())
                    shown: list[tuple[str, Any]] = items[:max_items]
                    if (
                        all(
                            not isinstance(val, dict) and not isinstance(val, list)
                            for _, val in shown
                        )
                        and len(shown) <= 2
                    ):
                        body = ", ".join(
                            f"{k!r}: {dict_repr(val, depth + 1)}" for k, val in shown
                        )
                        if len(items) > max_items:
                            body += ", …"
                        return "{" + body + "}"
                    lines: list[str] = []
                    for k, val in shown:
                        lines.append(
                            f"{pad}{' ' * indent}{k!r}: {dict_repr(val, depth + 1, max_depth, max_items, max_str, indent)}"
                        )
                    if len(items) > max_items:
                        lines.append(f"{pad}{' ' * indent}…")
                    return "{\n" + ",\n".join(lines) + f"\n{pad}" + "}"

                case list():
                    elems: list[Any] = v[:max_items]
                    if (
                        all(
                            isinstance(x, (int, float, str, bool, type(None)))
                            for x in elems
                        )
                        and len(elems) <= 4
                    ):
                        body = ", ".join(dict_repr(x, depth + 1) for x in elems)
                        if len(v) > max_items:
                            body += ", …"
                        return "[" + body + "]"
                    lines = [
                        f"{pad}{' ' * indent}{dict_repr(x, depth + 1, max_depth, max_items, max_str, indent)}"
                        for x in elems
                    ]
                    if len(v) > max_items:
                        lines.append(f"{pad}{' ' * indent}…")
                    return "[\n" + ",\n".join(lines) + f"\n{pad}" + "]"

                case str():
                    return repr(v if len(v) <= max_str else v[:max_str] + "…")
                case _:
                    return repr(v)

        return f"{self.__class__.__name__}({dict_repr(self.unwrap())})"

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

    def select(self: Dict[str, Any], *exprs: IntoExpr) -> Dict[str, Any]:
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

    def with_fields(self: Dict[str, Any], *exprs: IntoExpr) -> Dict[str, Any]:
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

    def equals_to(self, other: Self | Mapping[Any, Any]) -> bool:
        """
        Check if two records are equal based on their data.

        Args:
            other: Another Dict or mapping to compare against.

        Example:
        ```python
        >>> import pyochain as pc
        >>> d1 = pc.Dict({"a": 1, "b": 2})
        >>> d2 = pc.Dict({"a": 1, "b": 2})
        >>> d3 = pc.Dict({"a": 1, "b": 3})
        >>> d1.equals_to(d2)
        True
        >>> d1.equals_to(d3)
        False

        ```
        """
        other_data = other.unwrap() if isinstance(other, Dict) else other
        return self.unwrap() == other_data

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
