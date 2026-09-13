use crate::{
    abc,
    collections::sorted::{
        SortedItemsView, SortedKeysView, SortedValuesView, iter,
        traits::{ListGetter, Reduced, SortedCollectionsMethods, SortedDictMethods},
        views::{SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView},
    },
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*, types::PyDict};
use pyo3_ext::prelude::*;
use sorted_rs::{DictData, InnerGetter, KeysListsData, ListDataOwner, ListsData, ListsDataMethods};
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
    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let dict = self.get_dict(py).into_any();
        let items = self
            .try_lock()
            .inner()
            .iter()
            .map(|x| x.bind(py))
            .map(|key| {
                dict.get_item(key)
                    .and_then(|value| Ok(format!("{}: {}", key.repr()?, value.repr()?)))
            })
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("{type_name}({{{items}}})"))
    }
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
        py: Python<'_>,
        key: Py<PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let list = KeysListsData::new(key);
        let slf = DictData::new(list, PyDict::new(py).unbind()).conv::<Self>();
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }

    #[getter]
    fn get_key(&self, py: Python<'_>) -> Py<PyAny> {
        self.try_lock().list().2.clone_ref(py)
    }

    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(key)
    }

    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(key)
    }
}
impl SortedDictMethods for SortedKeyDict {
    type L = KeysListsData;
    type IView = SortedByKeyItemsView;
    type KView = SortedByKeyKeysView;
    type VView = SortedByKeyValuesView;
    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.try_lock();
        let type_name = Self::type_object(py).name()?;
        let key_arg = format!("{}, ", inner.list().2.bind(py).repr()?);
        let dict = self.get_dict(py).into_any();
        let items = self
            .try_lock()
            .inner()
            .iter()
            .map(|key| dict.get_item(key).map(|value| format!("{key}: {value}")))
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("{type_name}({key_arg}{{{items}}})"))
    }
}
impl SortedCollectionsMethods for SortedDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self.get_dict(py).copy().and_then(|x| tuple!(x))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(value)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
}

impl SortedCollectionsMethods for SortedKeyDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self
            .get_dict(py)
            .copy()
            .and_then(|x| tuple!(x.as_any(), self.try_lock().list().2.bind(py)))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.try_lock().contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.try_lock();
        let key = data.list().2.bind(py).call1((value,))?;
        data.list_mut().bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.try_lock();
        let key = data.list().2.bind(py).call1((value,))?;
        data.list_mut().bisect_right(&key)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
}
