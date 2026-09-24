use pyo3::{
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
use sorted_rs::{
    KeysListsData, ListsData, PySetDataRef, SetData, SetOp, SetPred, types::IntOrSlice,
};
use std::sync::{Arc, RwLock};
use std_tools::prelude::ResultExt;
use tap::{Conv, Pipe};
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSet)]
pub struct SortedSet(pub(super) Arc<RwLock<SetData<ListsData>>>);
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
pub struct SortedKeySet(pub(super) Arc<RwLock<SetData<KeysListsData>>>);
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
impl PySetDataRef for SortedSet {
    type L = ListsData;
}
impl PySetDataRef for SortedKeySet {
    type L = KeysListsData;
}
#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait SortedSetMethods:
    SortedCollectionsMethods
    + ListGetter<T = SetData<<Self as ListGetter>::L>>
    + IntoInit
    + PySetDataRef<L = <Self as ListGetter>::L>
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
        self.write()
            .get_item_or_slice(index)
            .and_then_left(Bound::try_into_py)
    }
    fn __delitem__(&self, index: IntOrSlice<'_>) -> PyResult<()> {
        self.write().del_item_or_slice(index)
    }
    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Eq)
    }
    fn __ne__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Ne)
    }
    fn __lt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Lt)
    }
    fn __gt__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Gt)
    }
    fn __le__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Le)
    }
    fn __ge__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        self.comp(other, CompareOp::Ge)
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
        self.map_any_mut(other, SetOp::Difference)
    }
    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::Intersection)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, SetOp::Intersection)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, SetOp::Union)
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
        self.write().add(value)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.write().discard(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.lock().copy(py)?.conv::<Self>().into_bound(py)
    }
    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.map_pred(other, SetPred::Disjoint)
    }
    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.map_pred(other, SetPred::Subset)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.map_pred(other, SetPred::Superset)
    }
    fn clear(&self, py: Python<'_>) {
        self.write().clear(py);
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.lock().count(value)
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.write().pop(py, index)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Difference)
    }
    #[pyo3(signature = (*iterables))]
    fn difference_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, SetOp::Difference)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Intersection)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, SetOp::Intersection)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.write().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, SetOp::SymmetricDifference)
    }
    fn symmetric_difference_update(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, SetOp::SymmetricDifference)
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_tup_into_bound(iterables, SetOp::Union)
    }
    #[pyo3(signature = (*iterables))]
    fn update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, SetOp::Union)
    }
    #[skip]
    #[inline]
    fn map_tup_into_bound<'py>(
        &self,
        tup: Bound<'py, PyTuple>,
        op: SetOp,
    ) -> PyResult<Bound<'py, Self>> {
        let py = tup.py();
        self.map_iter(tup, op)?.into_bound(py)
    }
    #[skip]
    #[inline]
    fn map_into_bound<'py>(
        &self,
        other: Bound<'py, PyAny>,
        op: SetOp,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.map_any(other, op)?.into_bound(py)
    }
}
