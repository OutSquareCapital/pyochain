use crate::abc;
use derive_more::From;
use pyo3::prelude::*;
use sorted_rs::{DictData, KeysListsData, ListsData, SetData, iter::Bounded};

macro_rules! impl_sorted_iter {
    ($($t:ty => $name:ident),+ $(,)?) => {
        $(
            #[derive(From)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $name(Bounded<$t>);
            #[pymethods]
            impl $name {
                fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    self.0.next(py)
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
