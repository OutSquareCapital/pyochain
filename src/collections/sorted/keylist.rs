use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{ListGetter, Reduced, SortedCollectionsMethods, SortedListMethods},
    },
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*};
use pyo3_ext::prelude::*;
use sorted_rs::{InnerGetter, KeysListsData, ListDataOwner, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList(pub(super) Arc<Mutex<KeysListsData>>);
impl ListGetter for SortedKeyList {
    type T = KeysListsData;
    type I = iter::PyBoundedKey;
    type IRev = iter::PyBoundedKeyRev;
    type IFull = iter::PyFullKey;
    type IFullRev = iter::PyFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
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
impl SortedCollectionsMethods for SortedKeyList {
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
        let key = data.list_mut().2.bind(value.py()).call1((value,))?;
        data.list_mut().bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut data = self.try_lock();
        let key = data.list_mut().2.bind(value.py()).call1((value,))?;
        data.list_mut().bisect_right(&key)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
}
impl From<KeysListsData> for SortedKeyList {
    fn from(data: KeysListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
