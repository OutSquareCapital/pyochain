use std::ptr::NonNull;

use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyList, PySet, PyTuple},
};
use pyo3_ext::prelude::*;

use tap::Pipe;

use crate::{
    abc,
    collections::sorted::{
        SortedList,
        traits::{
            BaseSortedListSet, BaseSortedSet, IntoUpdate, Reduced, SortedCollection,
            SortedListGetters, update_sorted_set,
        },
    },
    traits::PyoABC,
};

#[pyclass(module = "pyochain._collections", frozen, generic, extends = abc::PyoMutableSet)]
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
    fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        abc::PyoMutableSet::build_init()
            .add_subclass(self)
            .pipe(|x| Bound::new(py, x))
    }
}
#[pymethods]
impl SortedSet {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let init = Self::new(PySet::empty(py).unwrap(), SortedList::new());
        if let Some(iterable) = iterable {
            update_sorted_set(&init, py, IntoUpdate::from_any(iterable))?;
        }
        abc::PyoMutableSet::build_init().add_subclass(init).pipe(Ok)
    }
}
impl SortedCollection for SortedSet {
    fn __contains__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        self.set.bind(value.py()).contains(value)
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        PyTuple::new(py, [self.set.clone_ref(py)]).map(|tup| (Self::type_object(py), tup))
    }

    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_list().bisect_right(value)
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        self.get_list().index(value, start, stop)
    }
    #[allow(unused_variables)]
    fn islice<'py>(
        slf: Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        todo!()
    }
    #[allow(unused_variables)]
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        todo!()
    }

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.get_list().reset(py, load)
    }
    fn clear(&self) -> () {
        // Safety: This is what happen in fact inside every `.bind(py)` call.
        // We use it here to avoid changing the trait method signature just for SortedSet.
        let set: &Bound<'_, PySet> = unsafe { NonNull::from(&self.set).cast().as_ref() };
        set.clear();
        self.get_list().clear()
    }
}
impl BaseSortedListSet for SortedSet {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let set = self.set.bind(py);
        if !set.contains(&value)? {
            set.add(&value)?;
            self.get_list().add(py, value)?;
        }
        Ok(())
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let set = self.set.bind(value.py());
        if set.contains(&value)? {
            set.remove(&value)?;
            self.get_list().remove(value)?;
        }
        Ok(())
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.set.bind(value.py()).remove(&value)?;
        self.get_list().remove(value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        PySet::new(py, self.set.bind(py).iter()).and_then(|x| self.from_set(x))
    }
}
impl BaseSortedSet for SortedSet {
    type T = SortedList;
    #[inline(always)]
    fn get_list(&self) -> &SortedList {
        &self.list.get()
    }
    #[inline(always)]
    fn get_set(&self) -> &Py<PySet> {
        &self.set
    }
    fn from_vec<'py>(&self, py: Python<'py>, v: Vec<Py<PyAny>>) -> PyResult<Bound<'py, Self>> {
        SortedList::from_vec(py, v)
            .map(|list| Self::new(PySet::empty(py).unwrap(), list))
            .and_then(|x| x.into_bound(py))
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
            .get_data()
            .iter()
            .collect_bound::<PyList>(py)?
            .repr()?;
        Ok(format!("{}({})", type_name, self_repr))
    }
}
