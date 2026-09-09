use crate::{
    abc,
    collections::sorted::traits::{BaseSortedList, ListGetter, Reduced, SortedCollection},
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*};
use pyo3_ext::prelude::*;
use sorted_rs::{InnerGetter, KeysListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList(pub(super) Arc<Mutex<KeysListsData>>);
#[pymethods]
impl SortedKeyList {
    #[new]
    #[pyo3(signature = (key, iterable = None, /))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = key.unbind().pipe(KeysListsData::new).conv::<Self>();
        if let Some(iterable) = iterable {
            slf.extend(&iterable)?;
        }
        slf.init().pipe(Ok)
    }

    pub(super) fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_left(key)
    }
    pub(super) fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_right(key)
    }
}
impl SortedCollection for SortedKeyList {
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().contains(value)
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let data = self.try_lock();
        data.inner()
            .as_pylist(py)
            .and_then(|x| tuple!(x.as_any(), data.2.bind(py)))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut data = self.try_lock();
        let key = data.2.bind(value.py()).call1((value,))?;
        data.bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
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
    ) -> PyResult<usize> {
        self.try_lock().index(&value, start, stop)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().reset(py, load)
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
            data.inner().repeat(py, 2)
        } else {
            data.inner().concat(py, other)?
        };
        KeysListsData::from_vec(py, out, data.2.clone_ref(py))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let data = self.try_lock();
        KeysListsData::from_vec(py, data.inner().collapse(py), data.2.clone_ref(py))
            .map(Self::from)?
            .into_bound(py)
    }
}
impl From<KeysListsData> for SortedKeyList {
    fn from(data: KeysListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
