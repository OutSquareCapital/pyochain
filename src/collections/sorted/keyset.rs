use pyo3::{
    IntoPyObjectExt, PyTypeInfo,
    prelude::*,
    types::{PyList, PySet},
};
use pyo3_ext::prelude::CollectBoundIterator;
use sorted_rs::{KeysListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::Pipe;

use crate::{
    abc,
    collections::sorted::traits::{BaseSortedSet, IntoUpdate, ListGetter, PyIdentity},
    traits::IntoInit,
};
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedKeySet(pub(super) Arc<Mutex<KeysListsData>>, Py<PySet>, Py<PyAny>);
impl SortedKeySet {
    fn new(set: Bound<'_, PySet>, list: KeysListsData, key: Py<PyAny>) -> Self {
        let list = list.pipe(Mutex::new).pipe(Arc::new);
        Self(list, set.unbind(), key)
    }
}
#[pymethods]
impl SortedKeySet {
    #[new]
    #[pyo3(signature = (iterable = None, key = None))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let key_fn = key.map_or_else(|| PyIdentity.into_py_any(py).unwrap(), Bound::unbind);
        let list = KeysListsData::new(key_fn.clone_ref(py));
        let slf = Self::new(PySet::empty(py).unwrap(), list, key_fn);

        if let Some(iterable) = iterable {
            slf.update(py, IntoUpdate::from_any(iterable))?;
        }
        slf.init().pipe(Ok)
    }
    #[getter]
    fn get_key<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.2.bind(py)
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_left(key)
    }

    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_right(key)
    }
}
impl BaseSortedSet for SortedKeySet {
    #[inline(always)]
    fn get_set(&self) -> &Py<PySet> {
        &self.1
    }
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list =
            KeysListsData::from_vec(py, values.iter().map(Bound::unbind).collect(), &self.2)?;
        Self::new(values, list, self.2.clone_ref(py)).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let key = format!(", key={}", self.2.bind(py).repr()?);
        let type_name = Self::type_object(py).name()?;
        let list_repr = self.get_data().iter().collect_bound::<PyList>(py)?.repr()?;
        Ok(format!("{type_name}({list_repr}{key})"))
    }
}
