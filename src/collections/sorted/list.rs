use crate::{
    abc,
    collections::sorted::{
        iter,
        traits::{ListGetter, SortedListMethods},
    },
    traits::IntoInit,
};
use pyo3::prelude::*;
use sorted_rs::ListsData;
use std::sync::{Arc, Mutex};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList(pub(super) Arc<Mutex<ListsData>>);
impl ListGetter for SortedList {
    type T = ListsData;
    type I = iter::PyBounded;
    type IRev = iter::PyBoundedRev;
    type IFull = iter::PyFull;
    type IFullRev = iter::PyFullRev;
    #[inline(always)]
    fn inner(&self) -> &Arc<Mutex<Self::T>> {
        &self.0
    }
}
impl SortedListMethods for SortedList {
    type L = ListsData;
}
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
impl From<ListsData> for SortedList {
    fn from(data: ListsData) -> Self {
        Self(Arc::new(Mutex::new(data)))
    }
}
