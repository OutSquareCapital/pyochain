use pyo3::{PyTypeInfo, prelude::*, types::PySet};

use sorted_rs::{InnerGetter, KeysListsData, ListDataOwner, ListsData, SetData};
use std::sync::{Arc, Mutex};
use tap::{Conv, Pipe};

use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{ListGetter, SortedSetMethods},
    },
    traits::IntoInit,
};

#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet(pub(super) Arc<Mutex<SetData<ListsData>>>);
impl From<SetData<ListsData>> for SortedSet {
    fn from(data: SetData<ListsData>) -> Self {
        data.pipe(Mutex::new).pipe(Arc::new).pipe(Self)
    }
}
impl TryFrom<Bound<'_, PyAny>> for SortedSet {
    type Error = PyErr;
    fn try_from(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = iterable.py();
        let mut init = SetData::new(ListsData::default(), PySet::empty(py)?.unbind());
        init.update(iterable.try_into()?)?;
        Ok(init.into())
    }
}
impl ListGetter for SortedSet {
    type T = SetData<ListsData>;
    type I = iter::PySetBounded;
    type IRev = iter::PySetBoundedRev;
    type IFull = iter::PySetFull;
    type IFullRev = iter::PySetFullRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
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
        let mut inner = SetData::new(ListsData::default(), PySet::empty(py).unwrap().unbind());
        if let Some(iterable) = iterable {
            inner.update(iterable.try_into()?)?;
        }
        inner.conv::<Self>().init().pipe(Ok)
    }
}
impl SortedSetMethods for SortedSet {
    type L = ListsData;

    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let self_repr = self.lock().inner().as_pylist(py)?.repr()?;
        Ok(format!("{type_name}({self_repr})"))
    }
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedKeySet(pub(super) Arc<Mutex<SetData<KeysListsData>>>);
impl From<SetData<KeysListsData>> for SortedKeySet {
    fn from(data: SetData<KeysListsData>) -> Self {
        data.pipe(Mutex::new).pipe(Arc::new).pipe(Self)
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
        let mut inner = SetData::new(list, PySet::empty(py).unwrap().unbind());

        if let Some(iterable) = iterable {
            inner.update(iterable.try_into()?)?;
        }
        inner.conv::<Self>().init().pipe(Ok)
    }
}
impl ListGetter for SortedKeySet {
    type T = SetData<KeysListsData>;
    type I = iter::PySetBoundedKey;
    type IRev = iter::PySetBoundedKeyRev;
    type IFull = iter::PySetFullKey;
    type IFullRev = iter::PySetFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl SortedSetMethods for SortedKeySet {
    type L = KeysListsData;

    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.lock();
        let key = format!(", key={}", inner.list().2.bind(py).repr()?);
        let type_name = Self::type_object(py).name()?;
        let list_repr = inner.inner().as_pylist(py)?.repr()?;
        Ok(format!("{type_name}({list_repr}{key})"))
    }
}
