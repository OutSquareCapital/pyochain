use std::sync::RwLock;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{
    DictData, KeysListsData, ListsData, SetData,
    iter::{Bounded, Full},
};
use std_tools::prelude::*;
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
                    $method(&mut self.0.write_or_inner(), py)
                }
            }
        )+)+
    };
}

impl_sorted_iter! {
    ListsData => {
        Bounded => Bounded::next => PyBounded,
        Bounded => Bounded::next_back => PyBoundedRev,
        Full => Full::next => PyFull,
        Full => Full::next_back =>  PyFullRev,
    },
    KeysListsData => {
        Bounded  => Bounded::next => PyBoundedKey,
        Bounded  => Bounded::next_back => PyBoundedKeyRev,
        Full  =>Full::next=> PyFullKey,
        Full  => Full::next_back=> PyFullKeyRev,
    },
    SetData<ListsData> => {
        Bounded  => Bounded::next => PySetBounded,
        Bounded  => Bounded::next_back => PySetBoundedRev,
        Full => Full::next => PySetFull,
        Full => Full::next_back => PySetFullRev,
    },
    SetData<KeysListsData> => {
        Bounded  => Bounded::next => PySetBoundedKey,
        Bounded  => Bounded::next_back => PySetBoundedKeyRev,
        Full  => Full::next=> PySetFullKey,
        Full => Full::next_back => PySetFullKeyRev,
    },
    DictData<ListsData> => {
        Bounded  => Bounded::next => PyDictBounded,
        Bounded => Bounded::next_back => PyDictBoundedRev,
        Full => Full::next => PyDictFull,
        Full => Full::next_back => PyDictFullRev,
    },
    DictData<KeysListsData> => {
        Bounded  => Bounded::next=> PyDictBoundedKey,
        Bounded => Bounded::next_back => PyDictBoundedKeyRev,
        Full => Full::next => PyDictFullKey,
        Full  => Full::next_back=> PyDictFullKeyRev,
    },
}
