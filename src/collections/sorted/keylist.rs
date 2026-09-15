use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{ListGetter, SortedListMethods},
    },
    traits::IntoInit,
};
use pyo3::prelude::*;
use sorted_rs::KeysListsData;
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
impl SortedListMethods for SortedKeyList {
    type L = KeysListsData;
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
}
impl From<KeysListsData> for SortedKeyList {
    fn from(data: KeysListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
