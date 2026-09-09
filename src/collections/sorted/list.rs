use crate::{
    abc,
    collections::sorted::traits::{
        BaseSortedList, BaseSortedListSet, ListGetter, Reduced, SortedCollection,
    },
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*};
use pyo3_ext::prelude::*;
use sorted_rs::{InnerGetter, ListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList(pub(super) Arc<Mutex<ListsData>>);
impl SortedList {
    #[inline]
    pub(super) fn new() -> Self {
        Self(Arc::new(Mutex::new(ListsData::default())))
    }
    #[inline]
    fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Self> {
        let new_inst = Self::new();
        new_inst.try_lock().update(py, values)?;
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
        self.try_lock().contains(value)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        self.try_lock()
            .inner()
            .as_pylist(py)
            .and_then(|x| tuple!(x))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn clear(&self, _py: Python<'_>) {
        self.try_lock().clear();
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
}
impl BaseSortedListSet for SortedList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.try_lock().add(py, value)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().discard(value)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().remove(value)
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.try_lock().inner().collapse(py))?.into_bound(py)
    }
}
impl BaseSortedList for SortedList {
    fn __add__<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = slf.get().try_lock();
        let out = if other.is(&slf) {
            data.inner().repeat(py, 2)
        } else {
            data.inner().concat(py, other)?
        };
        Self::from_vec(py, out)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.try_lock().inner().repeat(py, num))?.into_bound(py)
    }

    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let cls_name = Self::type_object(py).name()?;
        self.try_lock()
            .inner()
            .as_pylist(py)?
            .repr()
            .map(|repr| format!("{cls_name}({repr})"))
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().count(&value)
    }
}
