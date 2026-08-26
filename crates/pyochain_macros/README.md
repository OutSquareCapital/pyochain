# pyochain-macros

This crate provides procedural macros for pyochain.

Just like [pyo3_ext](../pyo3_ext/README.md), this crate is NOT coupled with pyochain internals.

It notably provides a powerful `py_abc` macro to uses traits that are expanded in pyclasses, allowing to circumvent the limitations of pyclass single inheritance regarding code duplication and contract enforcement.
