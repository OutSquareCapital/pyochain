use crate::{
    abc,
    collections::sorted::{
        core::{ObjOrVec, SortedCollectionsMethods},
        getters::ListGetter,
    },
    traits::IntoInit,
};
use pyo3::{exceptions::PyNotImplementedError, prelude::*};
use pyo3_ext::{prelude::*, types::PyCmpOut};
use pyochain_macros::py_abc;
use sorted_rs::{
    KeysListsData, ListAdd, ListsData,
    prelude::*,
    types::{IntOrSlice, SeqOrAny},
};
use std::sync::{Arc, RwLock};
use std_tools::prelude::*;
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList(pub(super) Arc<RwLock<ListsData>>);

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

#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList(pub(super) Arc<RwLock<KeysListsData>>);

#[pymethods]
impl SortedKeyList {
    #[new]
    #[pyo3(signature = (key, iterable = None, /))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = key.unbind().pipe(KeysListsData::new).conv::<Self>();
        if let Some(iterable) = iterable {
            slf.extend(&iterable)?;
        }
        slf.init().pipe(Ok)
    }
}

#[py_abc(SortedList, SortedKeyList)]
pub(super) trait SortedListMethods:
    SortedCollectionsMethods
    + IntoInit
    + From<<Self as ListGetter>::L>
    + ListGetter<T = <Self as ListGetter>::L>
{
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let out = into_list_add(self, other)?;
        self.lock().concat(py, out)?.conv::<Self>().into_bound(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().eq(other)
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().ne(other)
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().lt(other)
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().gt(other)
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().le(other)
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.lock().ge(other)
    }

    fn __delitem__(&self, index: IntOrSlice<'_>) -> PyResult<()> {
        self.write().del_item_or_slice(index)
    }

    fn __getitem__<'py>(&self, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.write()
            .get_item_or_slice(index)
            .and_then_left(Bound::try_into_py)
    }
    fn __len__(&self) -> usize {
        self.lock().len
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__add__(other)
    }
    fn __rmul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.__mul__(py, num)
    }
    #[allow(unused_variables)]
    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``del sl[index]`` and ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    fn __iadd__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.extend(&other)
    }
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.lock()
            .as_repeated(py, num)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        self.write().imul(py, num)
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.lock().contains(value)
    }
    #[allow(unused_variables)]
    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.write().add(value)
    }
    fn clear(&self, py: Python<'_>) {
        self.write().clear(py);
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().count(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.write().discard(value)
    }
    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let out = into_list_add(self, iterable)?;
        match out {
            ListAdd::Identity => {
                let values = self.lock().collapse(py);
                self.write().extend(py, values)
            }
            ListAdd::Sorted(list) => {
                let values = list.collapse(py);
                self.write().extend(py, values)
            }
            ListAdd::Iterator(it) => {
                let values = it
                    .map(|x| x?.unbind().pipe(Ok))
                    .collect::<PyResult<Vec<_>>>()?;
                self.write().extend(py, values)
            }
        }
    }
    #[allow(unused_variables)]
    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.write().pop(py, index)
    }
    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write().remove(value)
    }
    fn reverse(&self) -> PyResult<()> {
        let msg = "use ``sl.rev()`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
}

fn into_list_add<'py, T: SortedListMethods>(
    left: &T,
    right: &'py Bound<'py, PyAny>,
) -> PyResult<ListAdd<'py, T::L>> {
    match right.cast_exact::<T>().map(Bound::get) {
        Ok(slf) if left.as_ref().is(slf.as_ref()) => ListAdd::Identity.pipe(Ok),
        Ok(list) => list.lock().pipe(ListAdd::Sorted).pipe(Ok),
        Err(_) => right.try_iter().map(ListAdd::Iterator),
    }
}
