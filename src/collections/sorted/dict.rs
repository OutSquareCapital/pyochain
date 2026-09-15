use crate::{
    abc,
    collections::sorted::{
        SortedItemsView, SortedKeysView, SortedValuesView, iter,
        traits::{ListGetter, SortedCollectionsMethods, SortedDictMethods},
        views::{SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView},
    },
    traits::IntoInit,
};
use pyo3::{prelude::*, types::PyDict};
use sorted_rs::{DictData, KeysListsData, ListDataOwner, ListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;

impl From<DictData<ListsData>> for SortedDict {
    fn from(data: DictData<ListsData>) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
impl From<DictData<KeysListsData>> for SortedKeyDict {
    fn from(data: DictData<KeysListsData>) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends= abc::PyoMutableMapping, mapping)]
pub struct SortedDict(pub(super) Arc<Mutex<DictData<ListsData>>>);

impl ListGetter for SortedDict {
    type T = DictData<ListsData>;
    type I = iter::PyDictBounded;
    type IRev = iter::PyDictBoundedRev;
    type IFull = iter::PyDictFull;
    type IFullRev = iter::PyDictFullRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
#[pymethods]
impl SortedDict {
    #[new]
    #[pyo3(signature = (iterable=None, **kwargs))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = DictData::<ListsData>::empty(py).conv::<Self>();
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }
}
impl SortedDictMethods for SortedDict {
    type L = ListsData;
    type IView = SortedItemsView;
    type KView = SortedKeysView;
    type VView = SortedValuesView;
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableMapping, mapping)]

pub struct SortedKeyDict(pub(super) Arc<Mutex<DictData<KeysListsData>>>);
impl ListGetter for SortedKeyDict {
    type T = DictData<KeysListsData>;
    type I = iter::PyDictBoundedKey;
    type IRev = iter::PyDictBoundedKeyRev;
    type IFull = iter::PyDictFullKey;
    type IFullRev = iter::PyDictFullKeyRev;
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
#[pymethods]
impl SortedKeyDict {
    #[new]
    #[pyo3(signature = (key, iterable=None, / , **kwargs))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let py = key.py();
        let list = KeysListsData::new(key.unbind());
        let slf = DictData::new(list, PyDict::new(py).unbind()).conv::<Self>();
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }
}
impl SortedDictMethods for SortedKeyDict {
    type L = KeysListsData;
    type IView = SortedByKeyItemsView;
    type KView = SortedByKeyKeysView;
    type VView = SortedByKeyValuesView;
}
impl SortedCollectionsMethods for SortedDict {
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect_right(value)
    }
}

impl SortedCollectionsMethods for SortedKeyDict {
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.lock();
        let key = data.list().2.bind(py).call1((value,))?;
        data.list_mut().bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.lock();
        let key = data.list().2.bind(py).call1((value,))?;
        data.list_mut().bisect_right(&key)
    }
}
