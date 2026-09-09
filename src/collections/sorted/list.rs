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
#[pymethods]
impl SortedList {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn py_new(iterable: Option<Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let data = Self::from(ListsData::default());
        if let Some(values) = iterable {
            data.update(&values)?;
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
        self.try_lock()
            .inner()
            .collapse(py)
            .pipe(|out| ListsData::from_vec(py, out))?
            .conv::<Self>()
            .into_bound(py)
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
        ListsData::from_vec(py, out)?.conv::<Self>().into_bound(py)
    }
}
impl From<ListsData> for SortedList {
    fn from(data: ListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
