use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{
            BaseSortedList, BaseSortedListSet, DEFAULT_LOAD_FACTOR, Reduced, SortedCollection,
            SortedListGetters,
        },
    },
    core::{PyoVec, iterators},
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*, types::PyList};
use pyo3_ext::prelude::*;
use sorted_rs::{Bounds, ListsData, ListsDataMethods};
use std::sync::{Mutex, atomic::AtomicUsize};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList {
    pub(super) data: Mutex<ListsData>,
    pub(super) load: AtomicUsize,
}
impl SortedList {
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            data: Mutex::new(ListsData::default()),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
        }
    }
    #[inline]
    pub(super) fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Self> {
        let new_inst = Self::new();
        new_inst
            .get_data()
            .update(py, values, new_inst.get_load())?;
        Ok(new_inst)
    }
}
#[pymethods]
impl SortedList {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn py_new(iterable: Option<Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let data = Self::new();
        if let Some(values) = iterable {
            data.py_update(&values)?;
        }

        data.init().pipe(Ok)
    }
}
impl SortedCollection for SortedList {
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_data().contains(value)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        self.get_data()
            .iter()
            .collect_bound::<PyList>(py)?
            .try_into_py::<PyoVec>()
            .and_then(|x| tuple!(x))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn clear(&self, _py: Python<'_>) {
        self.get_data().clear();
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_left(None, value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_right(None, value)
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.get_data().index(&value, start, stop)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.reset_list(py, load)
    }
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let specs = slf
            .get()
            .get_data()
            .pipe(|d| Bounds::get_irange_specs(&d.lists, &d.maxes, minimum, maximum, inclusive));

        match specs? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }

    fn islice(
        slf: Bound<'_, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        Self::islice_list(slf, start, stop, reverse)
    }
}
impl BaseSortedListSet for SortedList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.get_data().add(py, value, self.get_load())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.get_data().discard(value, self.get_load())
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.get_data().remove(value, self.get_load())
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().collapse(py))?.into_bound(py)
    }
}
impl BaseSortedList for SortedList {
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
        Self::from_vec(py, out)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().repeat(py, num))?.into_bound(py)
    }

    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let cls_name = Self::type_object(py).name()?;
        self.get_data()
            .iter()
            .collect_bound::<PyList>(py)?
            .repr()
            .map(|repr| format!("{cls_name}({repr})"))
    }

    fn wrap_iter(
        py: Python<'_>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'_, abc::PyoIterator>> {
        iter::SortedIter::new(inner)
            .into_bound(py)
            .map(Bound::into_super)
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.get_data().count(&value)
    }
}
