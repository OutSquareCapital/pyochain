use std::sync::Mutex;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{KeysListsData, ListDataIter, ListDataIterRev, ListDataIteratorMethods, ListsData};

macro_rules! impl_sorted_iter {
    ($name:ident, $iter:ty) => {
        #[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
        pub struct $name(Mutex<$iter>);
        impl $name {
            pub(super) fn new(inner: $iter) -> Self {
                Self(Mutex::new(inner))
            }
        }
        #[pymethods]
        impl $name {
            fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                self.0.lock().expect("poisoned").next(py)
            }
        }
    };
}

impl_sorted_iter!(SortedIter, ListDataIter<ListsData>);
impl_sorted_iter!(SortedIterReverse, ListDataIterRev<ListsData>);
impl_sorted_iter!(SortedIterKey, ListDataIter<KeysListsData>);
impl_sorted_iter!(SortedIterKeyReverse, ListDataIterRev<KeysListsData>);
