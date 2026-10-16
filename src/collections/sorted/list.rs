use crate::{
    abc,
    collections::sorted::{
        core::{ObjOrVec, SortedCollectionsMethods},
        getters::ListGetter,
    },
    traits::IntoInit,
};
use parking_lot::RwLock;
use pyo3::{exceptions::PyNotImplementedError, prelude::*};
use pyo3_ext::{prelude::*, types::PyCmpOut};
use pyochain_macros::py_abc;
use sorted_rs::{
    KeysListsData, ListsData, PyListsDataRef,
    prelude::*,
    types::{IntOrSlice, SeqOrAny},
};
use std_tools::prelude::*;
use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList(pub(super) RwLock<ListsData>);

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
pub struct SortedKeyList(pub(super) RwLock<KeysListsData>);

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
impl PyListsDataRef for SortedList {
    type L = ListsData;
}
impl PyListsDataRef for SortedKeyList {
    type L = KeysListsData;
}
#[py_abc(SortedList, SortedKeyList)]
pub(super) trait SortedListMethods:
    SortedCollectionsMethods
    + IntoInit
    + ListGetter<T = <Self as ListGetter>::L>
    + PyListsDataRef<L = <Self as ListGetter>::L>
{
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let out = self.try_as_updatable(other)?;
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
    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.write().append(value)
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
        self.write().extend(py, self.try_as_updatable(iterable)?)
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
