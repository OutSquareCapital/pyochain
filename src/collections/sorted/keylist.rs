use crate::{
    abc,
    collections::sorted::traits::{
        BaseSortedList, BaseSortedListSet, ListGetter, Reduced, SortedCollection,
    },
    core::PyoVec,
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*, types::PyList};
use pyo3_ext::prelude::*;
use sorted_rs::{KeysListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList(pub(super) Arc<Mutex<KeysListsData>>);
impl SortedKeyList {
    pub(super) fn new(key: Py<PyAny>) -> Self {
        Self(Arc::new(Mutex::new(KeysListsData::new(key))))
    }
    fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>, key: &Py<PyAny>) -> PyResult<Self> {
        let new_inst = Self::new(key.clone_ref(py));
        new_inst.try_lock().update(py, values)?;
        Ok(new_inst)
    }
}
#[pymethods]
impl SortedKeyList {
    #[new]
    #[pyo3(signature = (key, iterable = None, /))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = Self::new(key.unbind());
        if let Some(iterable) = iterable {
            slf.py_update(&iterable)?;
        }
        slf.init().pipe(Ok)
    }

    pub(super) fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.try_lock().bisect_left(key)
    }
    pub(super) fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.try_lock().bisect_right(key)
    }
}
impl SortedCollection for SortedKeyList {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let data = self.try_lock();
        data.iter()
            .collect_bound::<PyList>(py)?
            .try_into_py::<PyoVec>()
            .and_then(|x| tuple!(x.as_any(), data.2.bind(py)))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.try_lock();
        let key = data.2.bind(value.py()).call1((value,))?;
        data.bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.try_lock();
        let key = data.2.bind(value.py()).call1((value,))?;
        data.bisect_right(&key)
    }
    fn clear(&self, _py: Python<'_>) {
        self.try_lock().clear();
    }
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.try_lock().index(&value, start, stop)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().reset(py, load)
    }
}
impl BaseSortedListSet for SortedKeyList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.try_lock().add(py, value)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().discard(value)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().remove(value)
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let data = self.try_lock();
        Self::from_vec(py, data.collapse(py), &data.2)?.into_bound(py)
    }
}
impl BaseSortedList for SortedKeyList {
    fn __add__<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = slf.get().try_lock();
        let out = if other.is(&slf) {
            data.repeat(py, 2)
        } else {
            data.concat(py, other)?
        };
        Self::from_vec(py, out, &data.2)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        let data = self.try_lock();
        Self::from_vec(py, data.repeat(py, num), &data.2)?.into_bound(py)
    }

    //recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let data = self.try_lock();
        let key_repr = data.2.bind(py).repr()?;

        data.iter()
            .collect_bound::<PyList>(py)?
            .repr()
            .map(|repr| format!("{type_name}({repr}, key={key_repr})"))
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().count(&value)
    }
}
