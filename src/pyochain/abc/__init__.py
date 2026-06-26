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

from ._collection import PyoCollection, PyoSized
from ._mappings import PyoMapping, PyoMappingView, PyoMutableMapping
from ._sequences import PyoMutableSequence, PyoReversible, PyoSequence
from ._set import PyoMutableSet, PyoSet

from ._iterator import PyoIterable, PyoIterator, PyoContainer  # pyright: ignore[reportMissingModuleSource]

__all__ = [
    "Checkable",
    "Fluent",
    "Pipe",
    "PyoCollection",
    "PyoContainer",
    "PyoIterable",
    "PyoIterator",
    "PyoMapping",
    "PyoMappingView",
    "PyoMutableMapping",
    "PyoMutableSequence",
    "PyoMutableSet",
    "PyoReversible",
    "PyoSequence",
    "PyoSet",
    "PyoSized",
    "Tap",
]
