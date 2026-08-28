use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{
            BaseSortedList, BaseSortedListSet, PyIdentity, Reduced, SortedCollection,
            SortedListGetters,
        },
    },
    core::{PyoVec, iterators},
    traits::IntoInit,
};
use pyo3::{IntoPyObjectExt, PyTypeInfo, prelude::*, types::PyList};
use pyo3_ext::prelude::*;
use sorted_rs::{Bounds, KeysListsData, ListsDataMethods};
use std::sync::Mutex;
use tap::Pipe;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList(pub(super) Mutex<KeysListsData>);
impl SortedKeyList {
    pub(super) fn new(key: Py<PyAny>) -> Self {
        Self(Mutex::new(KeysListsData::new(key)))
    }
    pub(super) fn from_vec(
        py: Python<'_>,
        values: Vec<Py<PyAny>>,
        key: &Py<PyAny>,
    ) -> PyResult<Self> {
        let new_inst = Self::new(key.clone_ref(py));
        new_inst.get_data().update(py, values)?;
        Ok(new_inst)
    }
}
#[pymethods]
impl SortedKeyList {
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    pub(super) fn irange_key<'py>(
        slf: Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let data = slf.get().get_data();
        let specs = Bounds::get_irange_specs(&data.keys, &data.maxes, min_key, max_key, inclusive);
        match specs? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf.clone(), bounds, reverse),
        }
    }
    #[new]
    #[pyo3(signature = (iterable = None, *, key = None))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = Self::new(key.map_or_else(|| PyIdentity.into_py_any(py).unwrap(), Bound::unbind));

        if let Some(iterable) = iterable {
            slf.py_update(&iterable)?;
        }
        slf.init().pipe(Ok)
    }

    pub(super) fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_left(key)
    }
    pub(super) fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_right(key)
    }
}
impl SortedCollection for SortedKeyList {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        let data = self.get_data();
        data.iter()
            .collect_bound::<PyList>(py)?
            .try_into_py::<PyoVec>()
            .and_then(|x| tuple!(x.as_any(), data.key.bind(py)))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_data().contains(value)
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.get_data();
        let key = data.key.bind(value.py()).call1((value,))?;
        data.bisect_left(&key)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.get_data();
        let key = data.key.bind(value.py()).call1((value,))?;
        data.bisect_right(&key)
    }
    fn clear(&self, _py: Python<'_>) {
        self.get_data().clear();
    }
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.get_data().index(&value, start, stop)
    }
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let data = slf.get().get_data();
        let key_fn = |x| data.key.bind(slf.py()).call1((x,));
        let min_key = minimum.map(key_fn).transpose()?;
        let max_key = maximum.map(key_fn).transpose()?;
        Self::irange_key(slf.clone(), min_key, max_key, inclusive, reverse)
    }
    fn islice(
        slf: Bound<'_, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        Self::islice_list(slf, start, stop, reverse)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.get_data().reset(py, load)
    }
}
impl BaseSortedListSet for SortedKeyList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.get_data().add(py, value)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.get_data().discard(value)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.get_data().remove(value)
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let data = self.get_data();
        Self::from_vec(py, data.collapse(py), &data.key)?.into_bound(py)
    }
}
impl BaseSortedList for SortedKeyList {
    fn __add__<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = slf.get().get_data();
        let out = if other.is(&slf) {
            data.repeat(py, 2)
        } else {
            data.concat(py, other)?
        };
        Self::from_vec(py, out, &data.key)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        let data = self.get_data();
        Self::from_vec(py, data.repeat(py, num), &data.key)?.into_bound(py)
    }

    //recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let data = self.get_data();
        let key_repr = data.key.bind(py).repr()?;

        data.iter()
            .collect_bound::<PyList>(py)?
            .repr()
            .map(|repr| format!("{type_name}({repr}, key={key_repr})"))
    }

    fn wrap_iter(
        py: Python<'_>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        iter::SortedIterKey::new(inner)
            .into_bound(py)
            .map(Bound::into_super)
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.get_data().count(&value)
    }
}
