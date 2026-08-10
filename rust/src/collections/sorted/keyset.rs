use pyo3::{
    IntoPyObjectExt, PyTypeInfo,
    prelude::*,
    types::{PyList, PySet},
};
use pyo3_ext::prelude::*;
use tap::Pipe;

use crate::{
    abc,
    collections::{
        SortedKeyList,
        sorted::traits::{BaseSortedSet, IntoUpdate, PyIdentity, SortedListGetters},
    },
    traits::PyoABC,
};
#[pyclass(module = "pyochain._collections", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedKeySet {
    set: Py<PySet>,
    list: Py<SortedKeyList>,
    key: Py<PyAny>,
}
impl SortedKeySet {
    fn new(set: Bound<'_, PySet>, list: SortedKeyList, key: Py<PyAny>) -> Self {
        let py = set.py();
        let list = abc::PyoMutableSequence::build_init()
            .add_subclass(list)
            .pipe(|cls| Py::new(py, cls))
            .expect("Failed to create SortedKeyList instance from PyClassInitializer in SortedKeySet::new");
        Self {
            set: set.unbind(),
            list,
            key,
        }
    }

    fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        abc::PyoMutableSet::build_init()
            .add_subclass(self)
            .pipe(|x| Bound::new(py, x))
    }
}
#[pymethods]
impl SortedKeySet {
    #[new]
    #[pyo3(signature = (iterable = None, key = None))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let key_fn = key
            .map(Bound::unbind)
            .unwrap_or_else(|| PyIdentity.into_py_any(py).unwrap());
        let list = SortedKeyList::new(key_fn.clone_ref(py));
        let init = Self::new(PySet::empty(py).unwrap(), list, key_fn);

        if let Some(iterable) = iterable {
            init.update(py, IntoUpdate::from_any(iterable))?;
        }
        abc::PyoMutableSet::build_init().add_subclass(init).pipe(Ok)
    }
    #[getter]
    fn get_key<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.key.bind(py)
    }
    #[allow(unused_variables)]
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        slf: Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        slf.get()
            .get_list_bound(slf.py())
            .pipe(|list| SortedKeyList::irange_key(list, min_key, max_key, inclusive, reverse))
    }

    fn bisect_key_left(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_key_left(key)
    }

    fn bisect_key_right(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().get().bisect_key_right(key)
    }
}
impl BaseSortedSet for SortedKeySet {
    type T = SortedKeyList;
    #[inline(always)]
    fn get_list(&self) -> &Py<SortedKeyList> {
        &self.list
    }
    #[inline(always)]
    fn get_set(&self) -> &Py<PySet> {
        &self.set
    }
    fn from_set<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>> {
        let py = values.py();
        let list =
            SortedKeyList::from_vec(py, values.iter().map(Bound::unbind).collect(), &self.key)?;
        Self::new(values, list, self.key.clone_ref(py)).into_bound(py)
    }
    //@recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let key = format!(", key={}", self.key.bind(py).repr()?);
        let type_name = Self::type_object(py).name()?;
        let list_repr = self
            .list
            .get()
            .get_data()
            .iter()
            .collect_bound::<PyList>(py)?
            .repr()?;
        Ok(format!("{}({}{})", type_name, list_repr, key))
    }
}
