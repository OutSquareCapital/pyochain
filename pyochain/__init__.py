"""## pyochain: fluent Iterators, sorted containers, Result, Option, ABCs and more, written in Rust, for Python.

`pyochain` is a fast, open-source Python library for general-purpose programming with a fluent API.

Basic usage:

```python
from dataclasses import dataclass
from typing import override

from pyochain import NONE, Range, Seq, Some, Vec
from pyochain.abc import PyoIterable, PyoIterator, PyoMutableSequence, PyoSequence
from pyochain.collections import SortedList

type Person = tuple[str, int, str]


@dataclass
class Persons(PyoIterable[Person]):
    names: PyoSequence[str]
    ages: PyoSequence[int]
    cities: PyoMutableSequence[str]

    @override
    def __iter__(self) -> PyoIterator[Person]:
        return self.names.iter().zip(self.ages, self.cities)


persons = Persons(
    names=Seq("Alice", "Bob", "Charlie"),
    ages=Range(25, 100, 5).iter().take(3).collect(Vec),
    cities=SortedList(("New York", "London", "Tokyo")),
)


filtered = (
    persons
    .iter()
    .filter_map_star(lambda name, age, _city: Some(name) if age > 28 else NONE)
    .collect(Seq)
)
assert filtered == ("Bob", "Charlie")
john = (
    persons.names
    .iter()
    .map(str.lower)
    .find(lambda name: name == "john")
    .map(lambda name: f"Found {name.capitalize()}")
    .ok_or("Not found")
    .map_err(ValueError)
)
assert john.is_err()

cities_it = persons.cities.tap(lambda lst: lst.append("Geneva")).iter()
assert cities_it.next() == Some("Geneva")
assert cities_it.next() == Some("London")
assert cities_it.last() == "Tokyo"
```

Python API Documentation: https://outsquarecapital.github.io/pyochain/
Source Code: https://github.com/OutSquareCapital/pyochain
"""

from . import abc, collections
from .pyochain import (
    NONE,
    Dict,
    Err,
    Iter,
    Null,
    Ok,
    Option,
    OptionType,
    OptionUnwrapError,
    Peekable,
    Range,
    Result,
    ResultType,
    ResultUnwrapError,
    Seq,
    Set,
    SetMut,
    SliceView,
    Some,
    Vec,
    option,
    then_if_some,
    then_if_true,
)

__all__ = [
    "NONE",
    "Dict",
    "Err",
    "Iter",
    "Null",
    "Ok",
    "Option",
    "OptionType",
    "OptionUnwrapError",
    "Peekable",
    "Range",
    "Result",
    "ResultType",
    "ResultUnwrapError",
    "Seq",
    "Set",
    "SetMut",
    "SliceView",
    "Some",
    "Vec",
    "abc",
    "collections",
    "option",
    "then_if_some",
    "then_if_true",
]
