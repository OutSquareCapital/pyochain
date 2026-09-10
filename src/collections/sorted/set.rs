use pyo3::{PyTypeInfo, prelude::*, types::PySet};

use sorted_rs::{InnerGetter, KeysListsData, ListsData, ListsDataMethods, SetData, SetDataMethods};
use std::sync::{Arc, Mutex};
use tap::Pipe;

use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{ListGetter, SortedSetMethods},
    },
    traits::IntoInit,
};

#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet(pub(super) Arc<Mutex<SetData>>);
impl ListGetter for SortedSet {
    type T = SetData;
    type I = iter::PyBounded;
    type IRev = iter::PyBoundedRev;
    type IFull = iter::PyFull;
    type IFullRev = iter::PyFullRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl SortedSet {
    fn new(set: Bound<'_, PySet>, list: ListsData) -> Self {
        let list = SetData(list, set.into()).pipe(Mutex::new).pipe(Arc::new);
        Self(list)
    }

    pub fn from_iterable(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = iterable.py();
        let init = Self::new(PySet::empty(py).unwrap(), ListsData::default());
        init.update(py, iterable.into())?;
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
            slf.update(py, iterable.into())?;
        }
        slf.init().pipe(Ok)
    }
}
impl SortedSetMethods for SortedSet {
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.0.try_lock().unwrap().1.clone_ref(py).into_bound(py)
    }
    #[inline(always)]
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list = self
            .try_lock()
            .get_list()
            .as_owned_from(py, values.iter().map(Bound::unbind).collect())?;
        Self::new(values, list).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let self_repr = self.try_lock().inner().as_pylist(py)?.repr()?;
        Ok(format!("{type_name}({self_repr})"))
    }
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedKeySet(pub(super) Arc<Mutex<KeysListsData>>, Py<PySet>);
impl SortedKeySet {
    fn new(set: Bound<'_, PySet>, list: KeysListsData) -> Self {
        let list = list.pipe(Mutex::new).pipe(Arc::new);
        Self(list, set.unbind())
    }
}
#[pymethods]
impl SortedKeySet {
    #[new]
    #[pyo3(signature = (key, iterable = None, /))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = key.py();
        let key_fn = key.unbind();
        let list = KeysListsData::new(key_fn.clone_ref(py));
        let slf = Self::new(PySet::empty(py).unwrap(), list);

        if let Some(iterable) = iterable {
            slf.update(py, iterable.into())?;
        }
        slf.init().pipe(Ok)
    }
    #[getter]
    fn get_key(&self, py: Python<'_>) -> Py<PyAny> {
        self.try_lock().2.clone_ref(py)
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_left(key)
    }

    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_right(key)
    }
}
impl ListGetter for SortedKeySet {
    type T = KeysListsData;
    type I = iter::PyBoundedKey;
    type IRev = iter::PyBoundedKeyRev;
    type IFull = iter::PyFullKey;
    type IFullRev = iter::PyFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl SortedSetMethods for SortedKeySet {
    #[inline(always)]
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.1.clone_ref(py).into_bound(py)
    }
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list = self
            .try_lock()
            .as_owned_from(py, values.iter().map(Bound::unbind).collect())?;
        Self::new(values, list).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.try_lock();
        let key = format!(", key={}", inner.2.bind(py).repr()?);
        let type_name = Self::type_object(py).name()?;
        let list_repr = inner.inner().as_pylist(py)?.repr()?;
        Ok(format!("{type_name}({list_repr}{key})"))
    }
}
