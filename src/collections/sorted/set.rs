use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyList, PySet},
};
use pyo3_ext::prelude::*;

use tap::Pipe;

use crate::{
    abc,
    collections::sorted::{
        SortedList,
        traits::{BaseSortedSet, IntoUpdate, ListGetter, SortedListGetters},
    },
    traits::PyoABC,
};

#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet {
    set: Py<PySet>,
    list: Py<SortedList>,
}
impl SortedSet {
    fn new(set: Bound<'_, PySet>, list: SortedList) -> Self {
        let py = set.py();
        let list = abc::PyoMutableSequence::build_init()
            .add_subclass(list)
            .pipe(|cls| Py::new(py, cls))
            .expect(
                "Failed to create SortedList instance from PyClassInitializer in SortedSet::new",
            );
        Self {
            set: set.unbind(),
            list,
        }
    }

    pub fn from_iterable(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = iterable.py();
        let init = Self::new(PySet::empty(py).unwrap(), SortedList::new());
        init.update(py, IntoUpdate::from_any(iterable))?;
        Ok(init)
    }
    pub fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        abc::PyoMutableSet::build_init()
            .add_subclass(self)
            .pipe(|x| Bound::new(py, x))
    }
}
#[pymethods]
impl SortedSet {
    #[new]
    #[pyo3(signature = (iterable = None))]
    pub fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let init = Self::new(PySet::empty(py).unwrap(), SortedList::new());
        if let Some(iterable) = iterable {
            init.update(py, IntoUpdate::from_any(iterable))?;
        }
        abc::PyoMutableSet::build_init().add_subclass(init).pipe(Ok)
    }
}
impl ListGetter for SortedSet {
    type T = SortedList;
    #[inline(always)]
    fn get_list(&self) -> &Py<SortedList> {
        &self.list
    }
}
impl BaseSortedSet for SortedSet {
    #[inline(always)]
    fn get_set(&self) -> &Py<PySet> {
        &self.set
    }
    fn from_set<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list = SortedList::from_vec(py, values.iter().map(Bound::unbind).collect())?;
        Self::new(values, list).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let self_repr = self
            .get_list()
            .get()
            .get_data()
            .iter()
            .collect_bound::<PyList>(py)?
            .repr()?;
        Ok(format!("{}({})", type_name, self_repr))
    }
}
