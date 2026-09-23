use std::sync::Mutex;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{
    DictData, KeysListsData, ListsData, SetData,
    iter::{Bounded, BoundedRev, Full, FullRev, ListDataIteratorMethods},
};
use std_tools::prelude::*;
macro_rules! impl_sorted_iter {
    ($($t:ty => { $($iter:ident => $name:ident),+ $(,)? }),+ $(,)?) => {
        $($(
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
            pub struct $name(Mutex<$iter<$t>>);
            impl From<$iter<$t>> for $name {
                fn from(inner: $iter<$t>) -> Self {
                    Self(Mutex::new(inner))
                }
            }
            #[pymethods]
            impl $name {
                fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    self.0.try_into_inner().next(py)
                }
            }
        )+)+
    };
}

impl_sorted_iter! {
    ListsData => {
        Bounded => PyBounded,
        BoundedRev => PyBoundedRev,
        Full => PyFull,
        FullRev => PyFullRev,
    },
    KeysListsData => {
        Bounded => PyBoundedKey,
        BoundedRev => PyBoundedKeyRev,
        Full => PyFullKey,
        FullRev => PyFullKeyRev,
    },
    SetData<ListsData> => {
        Bounded => PySetBounded,
        BoundedRev => PySetBoundedRev,
        Full => PySetFull,
        FullRev => PySetFullRev,
    },
    SetData<KeysListsData> => {
        Bounded => PySetBoundedKey,
        BoundedRev => PySetBoundedKeyRev,
        Full => PySetFullKey,
        FullRev => PySetFullKeyRev,
    },
    DictData<ListsData> => {
        Bounded => PyDictBounded,
        BoundedRev => PyDictBoundedRev,
        Full => PyDictFull,
        FullRev => PyDictFullRev,
    },
    DictData<KeysListsData> => {
        Bounded => PyDictBoundedKey,
        BoundedRev => PyDictBoundedKeyRev,
        Full => PyDictFullKey,
        FullRev => PyDictFullKeyRev,
    },
}
