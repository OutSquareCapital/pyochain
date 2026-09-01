use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyList, PySet},
};

use pyo3_ext::prelude::CollectBoundIterator;
use sorted_rs::{ListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::Pipe;

use crate::{
    abc,
    collections::sorted::traits::{BaseSortedSet, IntoUpdate, ListGetter},
    traits::IntoInit,
};

#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet(pub(super) Arc<Mutex<ListsData>>, Py<PySet>);
impl SortedSet {
    fn new(set: Bound<'_, PySet>, list: ListsData) -> Self {
        let list = list.pipe(Mutex::new).pipe(Arc::new);
        Self(list, set.unbind())
    }

    pub fn from_iterable(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = iterable.py();
        let init = Self::new(PySet::empty(py).unwrap(), ListsData::default());
        init.update(py, IntoUpdate::from_any(iterable))?;
        Ok(init)
    }
}
#[pymethods]
impl SortedSet {
    #[new]
    #[pyo3(signature = (iterable = None))]
    pub fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = Self::new(PySet::empty(py).unwrap(), ListsData::default());
        if let Some(iterable) = iterable {
            slf.update(py, IntoUpdate::from_any(iterable))?;
        }
        slf.init().pipe(Ok)
    }
}
impl BaseSortedSet for SortedSet {
    #[inline(always)]
    fn get_set(&self) -> &Py<PySet> {
        &self.1
    }
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list = ListsData::from_vec(py, values.iter().map(Bound::unbind).collect())?;
        Self::new(values, list).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let self_repr = self.get_data().iter().collect_bound::<PyList>(py)?.repr()?;
        Ok(format!("{type_name}({self_repr})"))
    }
}
