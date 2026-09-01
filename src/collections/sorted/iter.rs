use std::sync::Mutex;

use crate::abc;
use pyo3::prelude::*;
use sorted_rs::{KeysListsData, ListDataIter, ListsData};

#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIter(Mutex<ListDataIter<ListsData>>);
impl SortedIter {
    pub(super) fn new(inner: ListDataIter<ListsData>) -> Self {
        Self(Mutex::new(inner))
    }
}
#[pymethods]
impl SortedIter {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0.lock().expect("poisoned").next(py)
    }
}

#[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
pub struct SortedIterKey(Mutex<ListDataIter<KeysListsData>>);
impl SortedIterKey {
    pub(super) fn new(inner: ListDataIter<KeysListsData>) -> Self {
        Self(Mutex::new(inner))
    }
}
#[pymethods]
impl SortedIterKey {
    fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.0.lock().expect("poisoned").next(py)
    }
}
