"""Public mixins traits for internal pyochain types, and custom user implementations.

Since `Pipeable` and `Checkable` depend only on Self for arguments, returns types and internal logic, they can be safely added to any already existing class to provide additional functionality.

`PyoIterable` is a more specific trait, equivalent to subclassing `abc.Iterable`, but with additional methods and requirements specific to pyochain.
"""

from ..rs import Checkable, Pipeable
from ._iterable import (
    PyoCollection,
    PyoItemsView,
    PyoIterable,
    PyoIterator,
    PyoKeysView,
    PyoMapping,
    PyoMappingView,
    PyoMutableMapping,
    PyoMutableSequence,
    PyoSequence,
    PyoSet,
    PyoValuesView,
)

__all__ = [
    "Checkable",
    "Pipeable",
    "PyoCollection",
    "PyoItemsView",
    "PyoIterable",
    "PyoIterator",
    "PyoKeysView",
    "PyoMapping",
    "PyoMappingView",
    "PyoMutableMapping",
    "PyoMutableSequence",
    "PyoSequence",
    "PyoSet",
    "PyoValuesView",
]
