"""Mixins and ABCs provided by pyochain.

The ABCs in this module are designed to replicate the `collections::abc` builtin library, with additional methods provided by pyochain.

Each ABC is prefixed by `Pyo` to avoid name conflicts with the builtin ABCs.

They have 3 purposes:

1. Provide a common interface and DRY implementation for pyochain concrete classes.
2. A type hierarchy for static type checking, duck typing, isinstance checks, and flexibility in function signatures.
3. Custom subclassing for users who want to implement their own collection types, or add pyochain functionality to existing classes.

The mixins are simple, implementation-agnostic classes that can be added to any existing class to provide additional functionnality.


"""

from ..rs import Checkable, Fluent, Pipe, Tap  # noqa: I001
from ._iterable import PyoIterable  # pyright: ignore[reportMissingModuleSource]
from ._collection import (  # pyright: ignore[reportMissingModuleSource]
    PyoContainer,
    PyoSized,
    PyoCollection,
)
from ._sequences import (  # pyright: ignore[reportMissingModuleSource]
    PyoReversible,
    PyoSequence,
    PyoMutableSequence,
)
from ._mappings import (  # pyright: ignore[reportMissingModuleSource]
    PyoKeysView,
    PyoMapping,
    PyoMutableMapping,
    PyoValuesView,
    PyoItemsView,
    PyoMappingView,
)
from ._sets import PyoSet, PyoMutableSet  # pyright: ignore[reportMissingModuleSource]
from ._iterator import (  # pyright: ignore[reportMissingModuleSource]
    PyoIterator,
)

__all__ = [
    "Checkable",
    "Fluent",
    "Pipe",
    "PyoCollection",
    "PyoContainer",
    "PyoItemsView",
    "PyoIterable",
    "PyoIterator",
    "PyoKeysView",
    "PyoMapping",
    "PyoMappingView",
    "PyoMutableMapping",
    "PyoMutableSequence",
    "PyoMutableSet",
    "PyoReversible",
    "PyoSequence",
    "PyoSet",
    "PyoSized",
    "PyoValuesView",
    "Tap",
]
