use crate::{
    abc,
    collections::sorted::traits::{
        ListGetter, Reduced, SortedCollectionsMethods, SortedListMethods,
    },
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*};
use pyo3_ext::prelude::*;
use sorted_rs::{InnerGetter, ListDataOwner, ListsData, ListsDataMethods};
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList(pub(super) Arc<Mutex<ListsData>>);
impl SortedListMethods for SortedList {}
#[pymethods]
impl SortedList {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn py_new(iterable: Option<Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let data = Self::from(ListsData::default());
        if let Some(values) = iterable {
            data.extend(&values)?;
        }
        data.init().pipe(Ok)
    }
}
impl SortedCollectionsMethods for SortedList {
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
        self.try_lock().list_mut().clear();
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(value)
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.try_lock().list_mut().index(&value, start, stop)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().list_mut().reset(py, load)
    }
}
impl From<ListsData> for SortedList {
    fn from(data: ListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
