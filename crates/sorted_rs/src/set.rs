use std::{
    cmp::Ordering,
    sync::{Arc, Mutex},
};

use crate::{
    KeysListsData, ListsData, SetData,
    prelude::*,
    types::{IntOrSlice, ListOrAny},
};
use either::Either;
use pyo3::{
    PyClass,
    basic::CompareOp,
    prelude::*,
    types::{PyBool, PyList, PyNotImplemented, PySet, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::try_cast_into;
use std_tools::prelude::*;
use tap::prelude::*;
pub trait PySetDataRef:
    Sync
    + PyClass<Frozen = pyo3::pyclass::boolean_struct::True>
    + AsRef<Arc<Mutex<SetData<Self::L>>>>
    + From<SetData<Self::L>>
{
    type L: ListsDataMethods;

    #[inline(always)]
    fn comp<'py>(&self, value: Bound<'py, PyAny>, op: CompareOp) -> PyCmpOut<'py, bool> {
        let py = value.py();
        let slf = self.as_ref().try_into_inner().get_set(py);
        try_cast_into! {
            match value {
                CaseExact::Self(sorted) if self.as_ref().is(sorted.get().as_ref()) => {
                    op.on_identity().pipe(Either::Left).pipe(Ok)
                }
                CaseExact::Self(sorted) => slf
                    .rich_compare_bool(sorted.get().as_ref().try_into_inner().1.bind(py), op)
                    .map(Either::Left),
                Case::PySet(pyset) => slf.rich_compare_bool(pyset, op).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    #[inline]
    fn map_any(&self, other: Bound<'_, PyAny>, op: SetOp) -> PyResult<Self> {
        let other_set = Args::from_any(self, other);
        self.as_ref()
            .try_into_inner()
            .map_set(other_set, op)
            .map(Self::from)
    }
    #[inline]
    fn map_any_mut(&self, other: Bound<'_, PyAny>, op: SetOp) -> PyResult<()> {
        let other_set = Args::from_any(self, other);
        self.as_ref().try_into_inner().update(other_set, op)
    }
    #[inline]
    fn map_pred<'py>(&self, other: Bound<'py, PyAny>, op: SetPred) -> PyResult<Bound<'py, PyBool>> {
        let py = other.py();
        let args = Args::from_any(self, other);
        op.call(self.as_ref().try_into_inner().1.bind(py), args)
    }
    #[inline]
    fn map_iter(&self, tuple: Bound<'_, PyTuple>, op: SetOp) -> PyResult<Self> {
        let args = Args::from_tuple(self, tuple);
        self.as_ref()
            .try_into_inner()
            .map_set(args, op)
            .map(Self::from)
    }
    #[inline(always)]
    fn map_iter_mut(&self, tuple: Bound<'_, PyTuple>, op: SetOp) -> PyResult<()> {
        let args = Args::from_tuple(self, tuple);
        self.as_ref().try_into_inner().update(args, op)
    }
}
impl SetData<ListsData> {
    pub fn build(py: Python<'_>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        Self::build_inner(py, ListsData::default(), iterable)
    }
}
impl TryFrom<Bound<'_, PyAny>> for SetData<ListsData> {
    type Error = PyErr;
    fn try_from(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::build_inner(iterable.py(), ListsData::default(), Some(iterable))
    }
}
impl SetData<KeysListsData> {
    pub fn build(key: Bound<'_, PyAny>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let py = key.py();
        let list = KeysListsData::new(key.unbind());
        Self::build_inner(py, list, iterable)
    }
}
impl<T: ListsDataMethods> SetData<T> {
    #[inline(always)]
    fn build_inner(
        py: Python<'_>,
        mut list: T,
        iterable: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let set = PySet::empty(py)?;
        if let Some(it) = iterable {
            it.try_iter()?
                .try_for_each(|value| try_add(&mut list, &set, value?))?;
        }
        Ok(Self(list, set.unbind()))
    }
    pub fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.1.clone_ref(py).into_bound(py)
    }
    fn wrap(&self, values: Bound<'_, PySet>) -> PyResult<Self> {
        let py = values.py();
        let list = self
            .0
            .as_owned_from(py, values.iter().map(Bound::unbind).collect())?;
        Self(list, values.unbind()).pipe(Ok)
    }
    pub fn clear(&mut self, py: Python<'_>) {
        self.0.clear(py);
        self.1.bind(py).clear();
    }
    pub fn reset(&mut self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.0.reset(py, load)
    }
    pub fn get_item_or_slice<'py>(&mut self, index: IntOrSlice<'py>) -> PyResult<ListOrAny<'py>> {
        let py = index.py();
        match index {
            Either::Right(slice) => self
                .get_slice(&slice)?
                .iter()
                .collect_bound::<PyList>(py)
                .map(Either::Left),
            Either::Left(index) => self.get_item(py, index.extract()?).map(Either::Right),
        }
    }

    pub fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        let py = index.py();
        match index {
            Either::Right(slice) => {
                let values = self
                    .0
                    .get_slice(&slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.1.bind(py).difference_update((values,))?;
                self.0.del_slice(&slice)
            }
            Either::Left(int) => {
                let idx = int.extract::<isize>()?;
                let value = self.0.get_item(py, idx)?;
                self.1.bind(py).remove(&value)?;
                self.0.del_item(py, idx)
            }
        }
    }

    pub fn __len__(&self, py: Python<'_>) -> usize {
        self.1.bind(py).len()
    }
    pub fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        try_add(&mut self.0, self.1.bind(value.py()), value)
    }
    pub fn discard(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let set = self.1.bind(value.py());
        match set.contains(value) {
            Ok(true) => {
                set.remove(value)?;
                self.0.remove(value)
            }
            Ok(false) => Ok(()),
            Err(err) => Err(err),
        }
    }
    pub fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        self.1.bind(py).copy().and_then(|x| self.wrap(x))
    }
    pub fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        match self.1.bind(value.py()).contains(value) {
            Ok(true) => Ok(1),
            Ok(false) => Ok(0),
            Err(e) => Err(e),
        }
    }

    pub fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let value = self.0.pop(py, index)?;
        self.1.bind(py).remove(&value)?;
        Ok(value)
    }
    pub fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.1.bind(value.py()).remove(value)?;
        self.0.remove(value)
    }
    #[inline(always)]
    fn map_set(&self, other: Args<'_>, op: SetOp) -> PyResult<Self> {
        op.call(self.1.bind(other.py()), other)
            .and_then(|x| self.wrap(x))
    }
    #[inline(always)]
    fn update(&mut self, other: Args<'_>, op: SetOp) -> PyResult<()> {
        let py = other.py();
        let set = &self.1.clone_ref(py).into_bound(py);
        op.call_mut(set, other)?;
        self.0.clear(py);
        self.0.extend(py, set.iter().map(Bound::unbind).collect())
    }
}
#[derive(Clone, Copy)]
pub enum SetPred {
    Subset,
    Superset,
    Disjoint,
}
impl SetPred {
    #[inline(always)]
    fn call<'py>(self, set: &Bound<'py, PySet>, other: Args<'py>) -> PyResult<Bound<'py, PyBool>> {
        match (self, other) {
            (_, Args::Tuple(_)) => unreachable!(),
            (Self::Subset | Self::Superset, Args::None(py)) => Ok(PyBool::new(py, true).to_owned()),
            (Self::Disjoint, Args::None(py)) => Ok(PyBool::new(py, set.is_empty()).to_owned()),
            (Self::Subset, Args::Any(other)) => set.issubset(other),
            (Self::Superset, Args::Any(other)) => set.issuperset(other),
            (Self::Disjoint, Args::Any(other)) => set.isdisjoint(other),
        }
    }
}
pub enum Args<'py> {
    /// Either an identity reference between the two sets, or an empty tuple.
    None(Python<'py>),
    /// A tuple of arguments with length greater than 1.
    Tuple(Bound<'py, PyTuple>),
    /// Any other type of argument.
    Any(Bound<'py, PyAny>),
}
impl<'py> Args<'py> {
    fn py(&self) -> Python<'py> {
        match self {
            Self::Tuple(set) => set.py(),
            Self::Any(any) => any.py(),
            Self::None(py) => *py,
        }
    }
    #[inline(always)]
    fn from_tuple<C>(left: &C, tuple: Bound<'py, PyTuple>) -> Self
    where
        C: PySetDataRef,
    {
        let py = tuple.py();
        match tuple.len().cmp(&1) {
            Ordering::Less => Self::None(py),
            Ordering::Equal => Self::from_any(left, unsafe { tuple.get_item_unchecked(0) }),
            Ordering::Greater => tuple
                .into_iter()
                .map(|other| match other.cast_exact::<C>().map(Bound::get) {
                    Ok(other) if left.as_ref().is(other.as_ref()) => {
                        left.as_ref().try_into_inner().get_set(py).into_any()
                    }
                    Ok(other) => other.as_ref().try_into_inner().get_set(py).into_any(),
                    Err(_) => other,
                })
                .collect_bound(py)
                .expect("The chain tuple.iter().map().collect(tuple) should never fail, since the length isn't affected")
                .pipe(Self::Tuple),
        }
    }
    #[inline(always)]
    fn from_any<C>(left: &C, any: Bound<'py, PyAny>) -> Self
    where
        C: PySetDataRef,
    {
        let py = any.py();
        match any.cast_exact::<C>().map(Bound::get) {
            Ok(other) if left.as_ref().is(other.as_ref()) => Self::None(py),
            Ok(other) => Self::Any(other.as_ref().try_into_inner().get_set(py).into_any()),
            Err(_) => Self::Any(any),
        }
    }
}
#[derive(Clone, Copy)]
pub enum SetOp {
    Difference,
    Intersection,
    Union,
    SymmetricDifference,
}
impl SetOp {
    #[inline(always)]
    pub fn call<'py>(
        self,
        set: &Bound<'py, PySet>,
        other: Args<'py>,
    ) -> PyResult<Bound<'py, PySet>> {
        match (self, other) {
            (Self::Union | Self::Intersection, Args::None(_)) => set.copy(),
            (Self::Difference | Self::SymmetricDifference, Args::None(py)) => PySet::empty(py),
            (Self::Difference, Args::Any(any)) => set.difference((any,)),
            (Self::Intersection, Args::Any(any)) => set.intersection((any,)),
            (Self::Union, Args::Any(any)) => set.union((any,)),
            (Self::SymmetricDifference, Args::Any(any)) => set.symmetric_difference(any),
            (Self::Difference, Args::Tuple(tuple)) => set.difference(tuple),
            (Self::Intersection, Args::Tuple(tuple)) => set.intersection(tuple),
            (Self::Union, Args::Tuple(tuple)) => set.union(tuple),
            (Self::SymmetricDifference, Args::Tuple(tuple)) => set.symmetric_difference(tuple),
        }
    }
    #[inline(always)]
    pub fn call_mut<'py>(self, set: &Bound<'py, PySet>, other: Args<'py>) -> PyResult<()> {
        match (self, other) {
            (Self::Union | Self::Intersection, Args::None(_)) => Ok(()),
            (Self::Difference | Self::SymmetricDifference, Args::None(_)) => {
                set.clear();
                Ok(())
            }
            (Self::Difference, Args::Any(any)) => set.difference_update((any,)),
            (Self::Intersection, Args::Any(any)) => set.intersection_update((any,)),
            (Self::Union, Args::Any(any)) => set.update((any,)),
            (Self::SymmetricDifference, Args::Any(any)) => set.symmetric_difference_update(any),
            (Self::Difference, Args::Tuple(tuple)) => set.difference_update(tuple),
            (Self::Intersection, Args::Tuple(tuple)) => set.intersection_update(tuple),
            (Self::Union, Args::Tuple(tuple)) => set.update(tuple),
            (Self::SymmetricDifference, Args::Tuple(tuple)) => {
                set.symmetric_difference_update(tuple)
            }
        }
    }
}
#[inline(always)]
fn try_add<T>(list: &mut T, set: &Bound<'_, PySet>, value: Bound<'_, PyAny>) -> PyResult<()>
where
    T: ListsDataMethods,
{
    match set.contains(&value) {
        Ok(true) => Ok(()),
        Ok(false) => {
            set.add(&value)?;
            list.add(value)
        }
        Err(e) => Err(e),
    }
}
