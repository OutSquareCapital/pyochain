use parking_lot::RwLock;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{DictData, KeysListsData, ListsData, SetData, iter::Bounded};
macro_rules! impl_sorted_iter {
    ($($t:ty => { $($iter:ident => $method:expr => $name:ident),+ $(,)? }),+ $(,)?) => {
        $($(
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
            pub struct $name(RwLock<$iter<$t>>);
            impl From<$iter<$t>> for $name {
                fn from(inner: $iter<$t>) -> Self {
                    Self(RwLock::new(inner))
                }
            }
            #[pymethods]
            impl $name {
                fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    $method(&mut self.0.write(), py)
                }
            }
        )+)+
    };
}

impl_sorted_iter! {
    ListsData => {
        Bounded => Bounded::next => PyBounded,
        Bounded => Bounded::next_back => PyBoundedRev,
    },
    KeysListsData => {
        Bounded  => Bounded::next => PyBoundedKey,
        Bounded  => Bounded::next_back => PyBoundedKeyRev,
    },
    SetData<ListsData> => {
        Bounded  => Bounded::next => PySetBounded,
        Bounded  => Bounded::next_back => PySetBoundedRev,
    },
    SetData<KeysListsData> => {
        Bounded  => Bounded::next => PySetBoundedKey,
        Bounded  => Bounded::next_back => PySetBoundedKeyRev
    },
    DictData<ListsData> => {
        Bounded  => Bounded::next => PyDictBounded,
        Bounded => Bounded::next_back => PyDictBoundedRev
    },
    DictData<KeysListsData> => {
        Bounded  => Bounded::next=> PyDictBoundedKey,
        Bounded => Bounded::next_back => PyDictBoundedKeyRev
    },
}
