use parking_lot::RwLock;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{DictData, KeysListsData, ListsData, SetData, iter::Bounded};

macro_rules! impl_sorted_iter {
    ($($t:ty => $name:ident),+ $(,)?) => {
        $(
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $name(RwLock<Bounded<$t>>);

            impl From<Bounded<$t>> for $name {
                fn from(inner: Bounded<$t>) -> Self {
                    Self(RwLock::new(inner))
                }
            }
            #[pymethods]
            impl $name {
                fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    self.0.write().next(py)
                }
            }
        )+
    };
}

impl_sorted_iter! {
    ListsData => PyBounded,
    KeysListsData => PyBoundedKey,
    SetData<ListsData> => PySetBounded,
    SetData<KeysListsData> => PySetBoundedKey,
    DictData<ListsData> => PyDictBounded,
    DictData<KeysListsData> => PyDictBoundedKey,
}
