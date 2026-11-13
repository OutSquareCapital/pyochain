use crate::{
    abc,
    collections::sorted::{
        SortedItemsView, SortedKeysView, SortedValuesView,
        views::{
            PySortedViewMethods, SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView,
        },
    },
};
use derive_more::Deref;
use parking_lot::RwLock;
use pyo3::{
    prelude::*,
    types::{PyDict, PyMapping},
};
use pyo3_ext::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{KeyList, List, ListGetter, types};
use tap::prelude::*;

#[derive(Deref)]
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends= abc::PyoMutableMapping, mapping)]
pub struct SortedDict(pub(super) RwLock<types::Dict<List>>);

#[pymethods]
impl SortedDict {
    #[new]
    #[pyo3(signature = (iterable=None, **kwargs))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let mut dict = types::Dict::<List>::empty(py);
        dict.update(py, iterable, kwargs)?;
        dict.conv::<Self>().init().pipe(Ok)
    }
}
impl SortedDictMethods for SortedDict {
    type IView = SortedItemsView;
    type KView = SortedKeysView;
    type VView = SortedValuesView;
}
#[derive(Deref)]
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableMapping, mapping)]

pub struct SortedKeyDict(pub(super) RwLock<types::Dict<KeyList>>);

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
        let list = KeyList::new(key.unbind());
        let mut dict = types::Dict::new(list, PyDict::new(py).unbind());
        dict.update(py, iterable, kwargs)?;
        dict.conv::<Self>().init().pipe(Ok)
    }
}
impl SortedDictMethods for SortedKeyDict {
    type IView = SortedByKeyItemsView;
    type KView = SortedByKeyKeysView;
    type VView = SortedByKeyValuesView;
}

#[py_abc(SortedDict, SortedKeyDict)]
pub(super) trait SortedDictMethods:
    ListGetter<T = types::Dict<<Self as ListGetter>::L>> + From<types::Dict<Self::L>>
{
    type KView: PySortedViewMethods + From<Py<Self>>;
    type VView: PySortedViewMethods + From<Py<Self>>;
    type IView: PySortedViewMethods + From<Py<Self>>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.read().contains(value)
    }
    #[getter]
    fn get_dict<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        self.read().data.clone_ref(py).into_bound(py)
    }
    fn keys(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::KView>> {
        let py = slf.py();
        slf.unbind().conv::<Self::KView>().into_bound(py)
    }
    fn items(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::IView>> {
        let py = slf.py();
        slf.unbind().conv::<Self::IView>().into_bound(py)
    }
    fn values(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::VView>> {
        let py = slf.py();
        slf.unbind().conv::<Self::VView>().into_bound(py)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.read().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.read().__len__(py)
    }
    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.read().get_item(key)
    }
    fn __delitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write().del_item(key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.write().set_item(key, value)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other.py(), Some(other), None)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        self.read().or(value)?.conv::<Self>().into_bound(value.py())
    }
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        self.read().ror(value)?.conv::<Self>().into_bound(py)
    }
    fn clear(&self, py: Python<'_>) {
        self.write().clear(py);
    }
    #[staticmethod]
    #[pyo3(signature = (iterable, value = None, /))]
    fn from_keys<'py>(
        iterable: &Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, SortedDict>> {
        types::Dict::<List>::from_keys(iterable, value)?
            .conv::<SortedDict>()
            .into_bound(iterable.py())
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        &self,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.write().pop(key, default)
    }
    #[pyo3(signature = (index = -1))]
    fn popitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.write().popitem(py, index)
    }
    #[pyo3(signature = (index = -1))]
    fn peekitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        self.write().peekitem(py, index)
    }
    #[pyo3(signature = (key, default = None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.write().setdefault(key, default)
    }
    #[pyo3(signature = (m = None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        self.write().update(py, m, kwargs)
    }
}
