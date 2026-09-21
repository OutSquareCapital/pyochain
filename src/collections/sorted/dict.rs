use crate::{
    abc,
    collections::sorted::{
        SortedItemsView, SortedKeysView, SortedValuesView,
        core::SortedCollectionsMethods,
        getters::ListGetter,
        views::{
            SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView, SortedViewMethods,
        },
    },
    traits::IntoInit,
};
use pyo3::{
    prelude::*,
    types::{PyDict, PyMapping},
};
use pyochain_macros::py_abc;
use sorted_rs::{DictData, KeysListsData, ListsData, prelude::*};
use std::sync::{Arc, Mutex};
use tap::prelude::*;

#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends= abc::PyoMutableMapping, mapping)]
pub struct SortedDict(pub(super) Arc<Mutex<DictData<ListsData>>>);

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

#[py_abc(SortedDict, SortedKeyDict)]
pub(super) trait SortedDictMethods:
    SortedCollectionsMethods + ListGetter<T = DictData<Self::L>> + IntoInit + From<DictData<Self::L>>
where
    DictData<Self::L>: PyRepr,
{
    type L: ListsDataMethods;
    type KView: SortedViewMethods<L = Self::L>;
    type VView: SortedViewMethods<L = Self::L>;
    type IView: SortedViewMethods<L = Self::L>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.lock().contains(value)
    }
    #[getter]
    fn get_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.lock().get_dict().clone_ref(py).into_bound(py)
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::KView>> {
        self.as_ref().clone().conv::<Self::KView>().into_bound(py)
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::IView>> {
        self.as_ref().clone().conv::<Self::IView>().into_bound(py)
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self::VView>> {
        self.as_ref().clone().conv::<Self::VView>().into_bound(py)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.lock().__len__(py)
    }
    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.lock().get_item(key)
    }
    fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().del_item(key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().set_item(key, value)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other.py(), Some(other), None)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        self.lock().or(value)?.conv::<Self>().into_bound(value.py())
    }
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        self.lock().ror(value)?.conv::<Self>().into_bound(py)
    }
    fn clear(&self, py: Python<'_>) {
        self.lock().clear(py);
    }
    #[staticmethod]
    #[pyo3(signature = (iterable, value = None, /))]
    fn from_keys<'py>(
        iterable: &Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, SortedDict>> {
        DictData::<ListsData>::from_keys(iterable, value)?
            .conv::<SortedDict>()
            .into_bound(iterable.py())
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        &self,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.lock().pop(key, default)
    }
    #[pyo3(signature = (index = -1))]
    fn popitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.lock().popitem(py, index)
    }
    #[pyo3(signature = (index = -1))]
    fn peekitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.lock().peekitem(py, index)
    }
    #[pyo3(signature = (key, default = None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.lock().setdefault(key, default)
    }
    #[pyo3(signature = (m = None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.lock().update(py, m, kwargs)
    }
}
