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
use sorted_rs::{IntoUpdate, KeysListsData, ListsData, SetData, prelude::*, types::IntOrSlice};
use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};
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
impl SortedSetMethods for SortedSet {
    type L = ListsData;
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
impl SortedSetMethods for SortedKeySet {
    type L = KeysListsData;
}

#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait SortedSetMethods:
    Sync
    + PyClass<Frozen = pyo3::pyclass::boolean_struct::True>
    + SortedCollectionsMethods
    + ListGetter<T = SetData<Self::L>>
    + IntoInit
    + From<SetData<Self::L>>
    + PyTypeInfo
{
    type L: ListsDataMethods;
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
        self.map_into_bound(other, Self::T::difference)
    }
    fn __isub__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, Self::T::difference_update)
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, Self::T::intersection)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, Self::T::intersection_update)
    }

    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, Self::T::update)
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, Self::T::union)
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
        self.map_any(other, Self::T::is_disjoint)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.map_any(other, Self::T::is_subset)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.map_any(other, Self::T::is_superset)
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
        self.map_iter(iterables, Self::T::difference)
    }

    #[pyo3(signature = (*iterables))]
    fn difference_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, Self::T::difference_update)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_iter(iterables, Self::T::intersection)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, Self::T::intersection_update)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.map_into_bound(other, Self::T::symmetric_difference)
    }
    fn symmetric_difference_update(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.map_any_mut(other, Self::T::symmetric_difference_update)
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.map_iter(iterables, Self::T::union)
    }
    #[pyo3(signature = (*iterables))]
    fn update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.map_iter_mut(iterables, Self::T::update)
    }
    #[skip]
    #[inline]
    fn map_iter_mut<R, F: Fn(&mut Self::T, IntoUpdate<'_>) -> PyResult<R>>(
        &self,
        iterables: Bound<'_, PyTuple>,
        func: F,
    ) -> PyResult<R> {
        let py = iterables.py();
        match iterables.len().cmp(&1) {
            Ordering::Less => func(&mut self.lock(), IntoUpdate::SmallSet(PySet::empty(py)?)),
            Ordering::Equal => self.map_any_mut(unsafe { iterables.get_item_unchecked(0) }, func),
            Ordering::Greater => self.iter_into_set(iterables).and_then(|pyset| {
                let mut locked = self.lock();
                let update_set = IntoUpdate::from_sets(&locked.get_set(py), pyset);
                func(&mut locked, update_set)
            }),
        }
    }

    #[skip]
    #[inline]
    fn map_iter<'py, F: Fn(&Self::T, IntoUpdate<'_>) -> PyResult<Self::T>>(
        &self,
        iterables: Bound<'py, PyTuple>,
        func: F,
    ) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        match iterables.len().cmp(&1) {
            Ordering::Less => func(&mut self.lock(), IntoUpdate::SmallSet(PySet::empty(py)?)),
            Ordering::Equal => self.map_any(unsafe { iterables.get_item_unchecked(0) }, func),
            Ordering::Greater => self.iter_into_set(iterables).and_then(|pyset| {
                let mut locked = self.lock();
                let update_set = IntoUpdate::from_sets(&locked.get_set(py), pyset);
                func(&mut locked, update_set)
            }),
        }?
        .conv::<Self>()
        .into_bound(py)
    }
    #[skip]
    #[inline]
    fn map_any_mut<R, F: Fn(&mut Self::T, IntoUpdate<'_>) -> R>(
        &self,
        other: Bound<'_, PyAny>,
        func: F,
    ) -> R {
        let other_set = IntoUpdate::extract_from(self, other);
        func(&mut self.lock(), other_set)
    }
    #[skip]
    #[inline]
    fn map_any<'py, R, F: Fn(&Self::T, IntoUpdate<'py>) -> R>(
        &self,
        other: Bound<'py, PyAny>,
        func: F,
    ) -> R {
        let other_set = IntoUpdate::extract_from(self, other);
        func(&self.lock(), other_set)
    }
    #[skip]
    #[inline]
    fn map_into_bound<'py, F: Fn(&Self::T, IntoUpdate<'py>) -> PyResult<Self::T>>(
        &self,
        other: Bound<'py, PyAny>,
        func: F,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other_set = IntoUpdate::extract_from(self, other);
        func(&self.lock(), other_set)?.conv::<Self>().into_bound(py)
    }
    #[skip]
    #[inline]
    fn iter_into_set<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, PySet>> {
        let py = iterables.py();
        iterables
            .into_iter()
            .map(|other| IntoUpdate::extract_from(self, other))
            .try_fold(PySet::empty(py)?, |pyset, other| {
                match other {
                    IntoUpdate::SmallSet(set) | IntoUpdate::BigSet(set) => pyset.update((set,)),
                    IntoUpdate::Any(other) => pyset.update((other,)),
                }
                .map(|()| pyset)
            })
    }
}
