use crate::{
    abc,
    collections::sorted::{
        SortedItemsView, SortedKeysView, SortedValuesView,
        traits::{ListGetter, Reduced, SortedCollection, SortedDictMethods},
        views::{SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView},
    },
    traits::IntoInit,
};
use pyo3::{
    PyTypeInfo, ffi,
    prelude::*,
    types::{PyDict, PyMapping},
};
use pyo3_ext::prelude::*;
use sorted_rs::{InnerGetter, KeysListsData, ListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
/// Key-value pair type from a Python `Mapping`
type DictItem<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>);
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends= abc::PyoMutableMapping, mapping)]
pub struct SortedDict(pub(super) Arc<Mutex<ListsData>>, Py<PyDict>);
impl SortedDict {
    pub fn empty(py: Python<'_>) -> Self {
        let inner = PyDict::new(py);
        let list = ListsData::default().pipe(Mutex::new).pipe(Arc::new);
        Self(list, inner.unbind())
    }
    pub fn copy_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        &self,
        py: Python<'py>,
        v: I,
    ) -> PyResult<Self> {
        let inner = PyDict::new(py);
        let unbounded = v
            .into_iter()
            .map(|x| x.and_then(|(key, value)| fill_dict(&inner, py, key, value)))
            .map(|res| res.map(|(key, _)| key.unbind()))
            .collect::<PyResult<Vec<_>>>()?;

        let list = self
            .try_lock()
            .as_owned_from(py, unbounded)
            .map(Mutex::new)
            .map(Arc::new)?;
        Ok(Self(list, inner.unbind()))
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
        let slf = Self(
            ListsData::default().pipe(Mutex::new).pipe(Arc::new),
            PyDict::new(py).unbind(),
        );
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }
}
impl SortedDictMethods for SortedDict {
    type IView = SortedItemsView;
    type KView = SortedKeysView;
    type VView = SortedValuesView;
    fn get_dict(&self) -> &Py<PyDict> {
        &self.1
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.iter(py)
            .pipe(|v| self.copy_from_iter(py, v))?
            .into_bound(py)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = self.iter(py).chain(value.pipe(iter_mapping)?);
        self.copy_from_iter(py, items)?.into_bound(py)
    }

    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = value.pipe(iter_mapping)?.chain(self.iter(py));
        self.copy_from_iter(py, items)?.into_bound(py)
    }

    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let items = self
            .iter(py)
            .map(|x| x.and_then(|(k, v)| Ok(format!("{}: {}", k.repr()?, v.repr()?))))
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("{type_name}({{{items}}})"))
    }
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableMapping, mapping)]

pub struct SortedKeyDict(pub(super) Arc<Mutex<KeysListsData>>, Py<PyDict>);
impl SortedKeyDict {
    pub fn copy_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        &self,
        py: Python<'py>,
        v: I,
    ) -> PyResult<Self> {
        let inner = PyDict::new(py);
        let unbounded = v
            .into_iter()
            .map(|res| res.and_then(|(key, value)| fill_dict(&inner, py, key, value)))
            .map(|x| x.map(|(k, _)| k.unbind()))
            .collect::<PyResult<Vec<_>>>()?;
        let list = self
            .try_lock()
            .as_owned_from(py, unbounded)
            .map(Mutex::new)
            .map(Arc::new)?;
        Ok(Self(list, inner.unbind()))
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
        let list = KeysListsData::new(key).pipe(Mutex::new).pipe(Arc::new);
        let slf = Self(list, PyDict::new(py).unbind());
        slf.update(py, iterable, kwargs)?;
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
impl SortedDictMethods for SortedKeyDict {
    type IView = SortedByKeyItemsView;
    type KView = SortedByKeyKeysView;
    type VView = SortedByKeyValuesView;
    fn get_dict(&self) -> &Py<PyDict> {
        &self.1
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.iter(py)
            .pipe(|v| self.copy_from_iter(py, v))?
            .into_bound(py)
    }
    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.try_lock();
        let type_name = Self::type_object(py).name()?;
        let key_arg = format!("{}, ", inner.2.bind(py).repr()?);
        let dict = self.get_dict().bind(py).as_any();
        let items = self
            .try_lock()
            .inner()
            .iter()
            .map(|key| dict.get_item(key).map(|value| format!("{key}: {value}")))
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("{type_name}({key_arg}{{{items}}})"))
    }
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = value.pipe(iter_mapping)?.chain(self.iter(py));
        self.copy_from_iter(py, items)?.into_bound(py)
    }

    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = self.iter(py).chain(value.pipe(iter_mapping)?);
        self.copy_from_iter(py, items)?.into_bound(py)
    }
}
impl SortedCollection for SortedDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self.get_dict().bind(py).copy().and_then(|x| tuple!(x))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_dict().bind(value.py()).contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().bisect_right(value)
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

    fn clear(&self, py: Python<'_>) {
        self.get_dict().bind(py).clear();
        self.try_lock().clear();
    }
}

impl SortedCollection for SortedKeyDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self
            .get_dict()
            .bind(py)
            .copy()
            .and_then(|x| tuple!(x.as_any(), self.try_lock().2.bind(py)))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_dict().bind(value.py()).contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.try_lock();
        let key = data.2.bind(py).call1((value,))?;
        data.bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut data = self.try_lock();
        let key = data.2.bind(py).call1((value,))?;
        data.bisect_right(&key)
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

    fn clear(&self, py: Python<'_>) {
        self.get_dict().bind(py).clear();
        self.try_lock().clear();
    }
}
fn iter_mapping<'py>(
    mapping: &Bound<'py, PyMapping>,
) -> PyResult<impl Iterator<Item = PyResult<DictItem<'py>>>> {
    mapping
        .call_method0("items")?
        .try_iter()?
        .map(|iter| iter.and_then(|item| item.extract::<(Bound<'py, PyAny>, Bound<'py, PyAny>)>()))
        .pipe(Ok)
}
fn fill_dict<'py>(
    inner: &Bound<'py, PyDict>,
    py: Python<'py>,
    key: Bound<'py, PyAny>,
    value: Bound<'py, PyAny>,
) -> PyResult<DictItem<'py>> {
    match unsafe { ffi::PyDict_SetItem(inner.as_ptr(), key.as_ptr(), value.as_ptr()) } {
        -1 => Err(PyErr::fetch(py)),
        _ => Ok((key, value)),
    }
}
