use pyo3::{
    PyTypeInfo,
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
use pyochain_macros::{py_abc, try_cast_into};
use sorted_rs::{
    IntoUpdate, KeysListsData, ListsData, SetComp, SetData, prelude::*, types::IntOrSlice,
};
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
    SortedCollectionsMethods
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
    #[skip]
    #[inline]
    fn comp<'py>(&self, value: Bound<'py, PyAny>, op: CompareOp) -> PyCmpOut<bool, 'py> {
        let py = value.py();
        try_cast_into! {
            match value {
                CaseExact::Self(sorted) if self.is(sorted.get()) => SetComp::Identity,
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => {
                    SetComp::Comparable(self.get_set(py), sorted.get().get_set(py))
                }
                Case::PySet(pyset) => SetComp::Comparable(self.get_set(py), pyset),
                _ => SetComp::NotImplemented(py),
            }
        }
        .comp(op)
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
        let py = other.py();
        self.lock()
            .difference(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __isub__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update_any(other, Self::T::difference_update)
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.lock()
            .intersection(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update_any(other, Self::T::intersection_update)
    }

    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update_any(other, Self::T::update)
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.lock()
            .union(py, (other,))?
            .conv::<Self>()
            .into_bound(py)
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
        self.lock().is_disjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.lock().is_subset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.lock().is_superset(other)
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
        let py = iterables.py();
        self.lock()
            .difference(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }

    #[pyo3(signature = (*iterables))]
    fn difference_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.update_iter(iterables, Self::T::difference_update)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        self.lock()
            .intersection(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.update_iter(iterables, Self::T::intersection_update)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.lock().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        self.lock()
            .symmetric_difference(other)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn symmetric_difference_update(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update_any(other, Self::T::symmetric_difference_update)
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        self.lock()
            .union(py, iterables)?
            .conv::<Self>()
            .into_bound(py)
    }
    #[pyo3(signature = (*iterables))]
    fn update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<()> {
        self.update_iter(iterables, Self::T::update)
    }
    #[skip]
    #[inline]
    fn update_iter<F: Fn(&mut Self::T, IntoUpdate<'_>) -> PyResult<()>>(
        &self,
        iterables: Bound<'_, PyTuple>,
        func: F,
    ) -> PyResult<()> {
        let py = iterables.py();
        match iterables.len() {
            0 => Ok(()),
            1 => self.update_any(unsafe { iterables.get_item_unchecked(0) }, func),
            _ => iterables
                .into_iter()
                .map(|other| self.extract_set(other))
                .try_fold(PySet::empty(py)?, |pyset, other| match other {
                    IntoUpdate::SmallSet(other) | IntoUpdate::BigSet(other) => {
                        pyset.update((other,))?;
                        Ok(pyset)
                    }
                    IntoUpdate::Any(other) => {
                        pyset.update((other,))?;
                        Ok(pyset)
                    }
                })
                .and_then(|pyset| {
                    let mut locked = self.lock();
                    let update_set = IntoUpdate::from_sets(&locked.get_set(py), pyset);
                    func(&mut locked, update_set)
                }),
        }
    }
    #[skip]
    #[inline]
    fn update_any<F: Fn(&mut Self::T, IntoUpdate<'_>) -> PyResult<()>>(
        &self,
        other: Bound<'_, PyAny>,
        func: F,
    ) -> PyResult<()> {
        let other_set = self.extract_set(other);
        func(&mut self.lock(), other_set)
    }
    #[skip]
    #[inline]
    fn extract_set<'py>(&self, other: Bound<'py, PyAny>) -> IntoUpdate<'py> {
        let py = other.py();
        try_cast_into! {
            match other {
                CaseExact::Self(other) => {
                    let slf_set = self.get_set(py);
                    let other = other.get();
                    if self.is(other) {
                        IntoUpdate::SmallSet(slf_set)
                    } else {
                        IntoUpdate::from_sets(&slf_set, other.get_set(py))
                    }
                }
                CaseExact::PySet(pyset) => IntoUpdate::from_sets(&self.get_set(py), pyset),
                _ => IntoUpdate::Any(other),
            }
        }
    }
}
