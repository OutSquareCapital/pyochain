use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::{
            SortedItemsView, SortedKeysView, SortedValuesView,
            traits::{BaseSortedDict, ListGetter, Reduced, SortedCollection, SortedListGetters},
            views::{SortedByKeyItemsView, SortedByKeyKeysView, SortedByKeyValuesView},
        },
    },
    traits::IntoInit,
};
use pyo3::{
    PyTypeInfo, ffi,
    prelude::*,
    types::{PyDict, PyMapping},
};
use pyo3_ext::prelude::*;
use sorted_rs::ListsDataMethods;
use tap::prelude::*;
/// Key-value pair type from a Python `Mapping`
type DictItem<'py> = (Bound<'py, PyAny>, Bound<'py, PyAny>);
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends= abc::PyoMutableMapping, mapping)]
pub struct SortedDict {
    list: Py<SortedList>,
    inner: Py<PyDict>,
}
impl SortedDict {
    pub fn try_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        py: Python<'py>,
        v: I,
    ) -> PyResult<Self> {
        let inner = PyDict::new(py);
        let unbounded = v
            .into_iter()
            .map(|x| x.and_then(|(key, value)| fill_dict(&inner, py, key, value)))
            .map(|res| res.map(|(key, _)| key.unbind()))
            .collect::<PyResult<Vec<_>>>()?;

        let list = SortedList::from_vec(py, unbounded)?
            .into_bound(py)?
            .unbind();
        Ok(Self {
            list,
            inner: inner.unbind(),
        })
    }
}
impl ListGetter for SortedDict {
    type T = SortedList;
    fn get_list(&self) -> &Py<SortedList> {
        &self.list
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
        let slf = Self {
            list: SortedList::new().into_bound(py)?.unbind(),
            inner: PyDict::new(py).unbind(),
        };
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }
}
impl BaseSortedDict for SortedDict {
    type IView = SortedItemsView;
    type KView = SortedKeysView;
    type VView = SortedValuesView;
    fn get_inner(&self) -> &Py<PyDict> {
        &self.inner
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.iter(py)
            .pipe(|v| Self::try_from_iter(py, v))?
            .into_bound(py)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = self.iter(py).chain(value.pipe(iter_mapping)?);
        Self::try_from_iter(py, items)?.into_bound(py)
    }

    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = value.pipe(iter_mapping)?.chain(self.iter(py));
        Self::try_from_iter(py, items)?.into_bound(py)
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

pub struct SortedKeyDict {
    key: Py<PyAny>,
    list: Py<SortedKeyList>,
    inner: Py<PyDict>,
}
impl SortedKeyDict {
    pub fn try_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        py: Python<'py>,
        v: I,
        key: Py<PyAny>,
    ) -> PyResult<Self> {
        let inner = PyDict::new(py);
        let unbounded = v
            .into_iter()
            .map(|res| res.and_then(|(key, value)| fill_dict(&inner, py, key, value)))
            .map(|x| x.map(|(k, _)| k.unbind()))
            .collect::<PyResult<Vec<_>>>()?;

        let list = SortedKeyList::from_vec(py, unbounded, &key)?
            .into_bound(py)?
            .unbind();
        Ok(Self {
            key,
            list,
            inner: inner.unbind(),
        })
    }
}
#[pymethods]
impl SortedKeyDict {
    #[new]
    #[pyo3(signature = (iterable=None,*, key,  **kwargs))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        key: Py<PyAny>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let list = SortedKeyList::new(key.clone_ref(py))
            .into_bound(py)?
            .unbind();
        let slf = Self {
            key,
            list,
            inner: PyDict::new(py).unbind(),
        };
        slf.update(py, iterable, kwargs)?;
        slf.init().pipe(Ok)
    }

    #[getter]
    fn get_key(&self) -> &Py<PyAny> {
        &self.key
    }

    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        slf: &Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        SortedKeyList::irange_key(
            slf.get().get_list_bound(slf.py()),
            min_key,
            max_key,
            inclusive,
            reverse,
        )
    }

    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_key_left(key)
    }

    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_key_right(key)
    }
}
impl ListGetter for SortedKeyDict {
    type T = SortedKeyList;
    fn get_list(&self) -> &Py<SortedKeyList> {
        &self.list
    }
}
impl BaseSortedDict for SortedKeyDict {
    type IView = SortedByKeyItemsView;
    type KView = SortedByKeyKeysView;
    type VView = SortedByKeyValuesView;
    fn get_inner(&self) -> &Py<PyDict> {
        &self.inner
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.iter(py)
            .pipe(|v| Self::try_from_iter(py, v, self.key.clone_ref(py)))?
            .into_bound(py)
    }
    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let key_arg = format!("{}, ", self.key.bind(py).repr()?);
        let dict = self.get_inner().bind(py).as_any();
        let items = self
            .get_list()
            .get()
            .get_data()
            .iter()
            .map(|key| dict.get_item(key).map(|value| format!("{key}: {value}")))
            .collect::<PyResult<Vec<_>>>()?
            .join(", ");
        Ok(format!("{type_name}({key_arg}{{{items}}})"))
    }
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = value.pipe(iter_mapping)?.chain(self.iter(py));
        Self::try_from_iter(py, items, self.key.clone_ref(py))?.into_bound(py)
    }

    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = self.iter(py).chain(value.pipe(iter_mapping)?);
        Self::try_from_iter(py, items, self.key.clone_ref(py))?.into_bound(py)
    }
}
impl SortedCollection for SortedDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self.get_inner().bind(py).copy().and_then(|x| tuple!(x))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_inner().bind(value.py()).contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_right(value)
    }
    fn islice(
        slf: Bound<'_, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        SortedList::islice(slf.get().get_list_bound(slf.py()), start, stop, reverse)
    }

    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        SortedList::irange(
            slf.get().get_list_bound(slf.py()),
            minimum,
            maximum,
            inclusive,
            reverse,
        )
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.get_list().get().index(value, start, stop)
    }

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.get_list().get().reset(py, load)
    }

    fn clear(&self, py: Python<'_>) {
        self.get_inner().bind(py).clear();
        self.get_list().get().clear(py);
    }
}

impl SortedCollection for SortedKeyDict {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let items = self
            .get_inner()
            .bind(py)
            .copy()
            .and_then(|x| tuple!(x.as_any(), self.key.bind(py)))?;
        Ok((Self::type_object(py), items))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_inner().bind(value.py()).contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_right(value)
    }
    fn islice(
        slf: Bound<'_, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        SortedKeyList::islice(slf.get().get_list_bound(slf.py()), start, stop, reverse)
    }

    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        SortedKeyList::irange(
            slf.get().get_list_bound(slf.py()),
            minimum,
            maximum,
            inclusive,
            reverse,
        )
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.get_list().get().index(value, start, stop)
    }

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.get_list().get().reset(py, load)
    }

    fn clear(&self, py: Python<'_>) {
        self.get_inner().bind(py).clear();
        self.get_list().get().clear(py);
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
