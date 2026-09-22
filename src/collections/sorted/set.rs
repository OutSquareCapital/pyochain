use pyo3::{
    PyClass, PyTypeInfo,
    basic::CompareOp,
    prelude::*,
    types::{PyBool, PySet, PyTuple},
};

use super::{
    core::{ObjOrVec, SortedCollectionsMethods},
    getters::ListGetter,
};
use crate::{abc, traits::IntoInit};
use pyo3_ext::{prelude::*, types::PyCmpOut};
use pyochain_macros::py_abc;
use sorted_rs::{KeysListsData, ListsData, SetData, SetOp, SetPred, types::IntOrSlice};
use std::sync::{Arc, Mutex};
use std_tools::prelude::ResultExt;
use tap::{Conv, Pipe};
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet(pub(super) Arc<Mutex<SetData<ListsData>>>);
#[pymethods]
impl SortedSet {
    #[new]
    #[pyo3(signature = (iterable = None, /))]
    pub fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        SetData::<ListsData>::build(py, iterable)?
            .conv::<Self>()
            .init()
            .pipe(Ok)
    }
}
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedKeySet(pub(super) Arc<Mutex<SetData<KeysListsData>>>);
#[pymethods]
impl SortedKeySet {
    #[new]
    #[pyo3(signature = (key, iterable = None, /))]
    fn py_new(
        key: Bound<'_, PyAny>,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        SetData::<KeysListsData>::build(key, iterable)?
            .conv::<Self>()
            .init()
            .pipe(Ok)
    }
}

#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait SortedSetMethods:
    Sync
    + PyClass<Frozen = pyo3::pyclass::boolean_struct::True>
    + SortedCollectionsMethods
    + ListGetter<T = SetData<<Self as ListGetter>::L>>
    + IntoInit
    + From<SetData<<Self as ListGetter>::L>>
    + PyTypeInfo
{
    #[getter]
    #[inline(always)]
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.lock().get_set(py)
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_set(value.py()).contains(value)
    }
    fn __getitem__<'py>(&self, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.lock()
            .get_item_or_slice(index)
            .and_then_left(Bound::try_into_py)
    }
    fn __delitem__(&self, index: IntOrSlice<'_>) -> PyResult<()> {
        self.lock().del_item_or_slice(index)
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Eq)
    }

    fn __ne__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Ne)
    }

    fn __lt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Lt)
    }

    fn __gt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Gt)
    }

    fn __le__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Le)
    }

    fn __ge__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        Self::T::comp(self, other, CompareOp::Ge)
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.lock().__len__(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::Difference)
    }
    fn __isub__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        Self::T::map_any_mut(self, other, SetOp::Difference)
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::Intersection)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        Self::T::map_any_mut(self, other, SetOp::Intersection)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        Self::T::map_any_mut(self, other, SetOp::Union)
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::Union)
    }
    fn __ror__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(other)
    }
    fn __xor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __rxor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __ixor__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.symmetric_difference_update(other)
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().add(value)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().discard(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        Self::T::map_pred(self, other, SetPred::Disjoint)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        Self::T::map_pred(self, other, SetPred::Subset)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        Self::T::map_pred(self, other, SetPred::Superset)
    }
    fn clear(&self, py: Python<'_>) {
        self.lock().clear(py);
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.lock().count(value)
    }

    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.lock().pop(py, index)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Difference)
    }

    #[pyo3(signature = (*iterables))]
    fn difference_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        Self::T::map_iter_mut(self, iterables, SetOp::Difference)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Intersection)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        Self::T::map_iter_mut(self, iterables, SetOp::Intersection)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::SymmetricDifference)
    }
    fn symmetric_difference_update(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        Self::T::map_any_mut(self, other, SetOp::SymmetricDifference)
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Union)
    }
    #[pyo3(signature = (*iterables))]
    fn update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        Self::T::map_iter_mut(self, iterables, SetOp::Union)
    }
    #[skip]
    #[inline]
    fn map_tup_into_bound<'py>(
        &self,
        tup: Bound<'py, PyTuple>,
        op: SetOp,
    ) -> PyResult<Bound<'py, Self>> {
        let py = tup.py();
        Self::T::map_iter(self, tup, op)?.into_bound(py)
    }
    #[skip]
    #[inline]
    fn map_into_bound<'py>(
        &self,
        other: Bound<'py, PyAny>,
        op: SetOp,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        Self::T::map_any(self, other, op)?.into_bound(py)
    }
}
