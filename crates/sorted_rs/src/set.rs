use derive_more::Constructor;
use parking_lot::RwLock;
use std::cmp::Ordering;

use crate::{
    KeyList, List,
    prelude::*,
    slice::SliceIdxs,
    types::{IntOrSlice, ListOrAny, Set},
};
use either::Either;
use pyo3::{
    basic::CompareOp,
    prelude::*,
    types::{PyBool, PyNotImplemented, PySet, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::try_cast_into;
use tap::prelude::*;
pub trait PySetDataRef:
    PyFrozenClass + AsRef<RwLock<Set<Self::L>>> + From<Set<Self::L>> + Init
{
    type L: ListsMethods;

    #[inline(always)]
    fn comp<'py>(&self, value: Bound<'py, PyAny>, op: CompareOp) -> PyCmpOut<'py, bool> {
        let py = value.py();
        let slf = self.as_ref().read().get_set(py);
        try_cast_into! {
            match value {
                CaseExact::Self(sorted) if std::ptr::eq(sorted.get(), self) => {
                    op.on_identity().pipe(Either::Left).pipe(Ok)
                }
                CaseExact::Self(sorted) => slf
                    .rich_compare_bool(sorted.get().as_ref().read().1.bind(py), op)
                    .map(Either::Left),
                Case::PySet(pyset) => slf.rich_compare_bool(pyset, op).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    #[inline]
    fn op<'py, T>(&self, other: Bound<'py, T>, op: SetOp) -> PyResult<Bound<'py, Self>>
    where
        for<'a> Args<'py>: From<IntoArgs<'a, 'py, Self, T>>,
    {
        let py = other.py();
        let other_set = IntoArgs::new(self, other).conv::<Args<'py>>();
        let slf = self.as_ref().read();
        op.call(slf.1.bind(py), other_set)
            .and_then(|x| slf.wrap(x))?
            .conv::<Self>()
            .into_bound(py)
    }
    #[inline]
    fn op_mut<'py, T>(&self, other: Bound<'py, T>, op: SetOp) -> PyResult<()>
    where
        for<'a> Args<'py>: From<IntoArgs<'a, 'py, Self, T>>,
    {
        let py = other.py();
        let args = IntoArgs::new(self, other).conv::<Args<'_>>();
        let mut slf = self.as_ref().write();
        let set = &slf.1.clone_ref(py).into_bound(py);
        op.call_mut(set, args)?;
        slf.0.clear(py);
        slf.0.extend(py, set.iter_py().into())
    }
    #[inline]
    fn map_pred<'py>(&self, other: Bound<'py, PyAny>, op: SetPred) -> PyResult<Bound<'py, PyBool>> {
        let py = other.py();
        let args = IntoArgs::new(self, other).conv::<Args<'py>>();
        op.call(self.as_ref().read().1.bind(py), args)
    }
}
impl Set<List> {
    pub fn build(py: Python<'_>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        Self::build_inner(py, List::default(), iterable)
    }
}
impl TryFrom<Bound<'_, PyAny>> for Set<List> {
    type Error = PyErr;
    fn try_from(iterable: Bound<'_, PyAny>) -> PyResult<Self> {
        Self::build_inner(iterable.py(), List::default(), Some(iterable))
    }
}
impl Set<KeyList> {
    pub fn build(key: Bound<'_, PyAny>, iterable: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let py = key.py();
        let list = KeyList::new(key.unbind());
        Self::build_inner(py, list, iterable)
    }
}
impl<T: ListsMethods> Set<T> {
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
            Either::Right(pyslice) => SliceIdxs::new(&pyslice, self.len())?
                .extract_from(py, self)?
                .iter()
                .collect_bound(py)
                .map(Either::Left),
            Either::Left(index) => self.get_item(py, index.extract()?).map(Either::Right),
        }
    }

    pub fn del_item_or_slice(&mut self, index: IntOrSlice<'_>) -> PyResult<()> {
        let py = index.py();
        match index {
            Either::Right(pyslice) => {
                let slice = SliceIdxs::new(&pyslice, self.len())?;
                let values = slice
                    .extract_from(py, &mut self.0)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.1.bind(py).difference_update((values,))?;
                slice.delete_from(py, &mut self.0)
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
            (Self::Subset | Self::Superset, Args::None | Args::Slf) => {
                Ok(PyBool::new(set.py(), true).to_owned())
            }
            (Self::Disjoint, Args::None | Args::Slf) => {
                Ok(PyBool::new(set.py(), set.is_empty()).to_owned())
            }
            (Self::Subset, Args::Any(other)) => set.issubset(other),
            (Self::Superset, Args::Any(other)) => set.issuperset(other),
            (Self::Disjoint, Args::Any(other)) => set.isdisjoint(other),
        }
    }
}
#[derive(Constructor)]
pub struct IntoArgs<'a, 'py, C, T> {
    pyref: &'a C,
    obj: Bound<'py, T>,
}
pub enum Args<'py> {
    /// An empty tuple.
    None,
    /// An identity reference between the two sets used as operands
    Slf,
    /// A tuple of arguments with length greater than 1.
    Tuple(Bound<'py, PyTuple>),
    /// Any other type of argument.
    Any(Bound<'py, PyAny>),
}
impl<'py, C> From<IntoArgs<'_, 'py, C, PyTuple>> for Args<'py>
where
    C: PySetDataRef,
{
    #[inline(always)]
    fn from(args: IntoArgs<'_, 'py, C, PyTuple>) -> Self {
        let py = args.obj.py();
        match args.obj.len().cmp(&1) {
            Ordering::Less => Self::None,
            Ordering::Equal => IntoArgs::new(args.pyref, unsafe { args.obj.get_item_unchecked(0) }).into(),
            Ordering::Greater => args.obj
                .into_iter()
                .map(|other| match other.cast_exact::<C>().map(Bound::get) {
                    Ok(other) if std::ptr::eq(other, args.pyref) => {
                        args.pyref.as_ref().read().get_set(py).into_any()
                    }
                    Ok(other) => other.as_ref().read().get_set(py).into_any(),
                    Err(_) => other,
                })
                .collect_bound(py)
                .expect("The chain tuple.iter().map().collect(tuple) should never fail, since the length isn't affected")
                .pipe(Self::Tuple),
        }
    }
}

impl<'py, C> From<IntoArgs<'_, 'py, C, PyAny>> for Args<'py>
where
    C: PySetDataRef,
{
    #[inline(always)]
    fn from(args: IntoArgs<'_, 'py, C, PyAny>) -> Self {
        let py = args.obj.py();
        match args.obj.cast_exact::<C>().map(Bound::get).ok() {
            Some(other) if std::ptr::eq(other, args.pyref) => Self::Slf,
            Some(other) => Self::Any(other.as_ref().read().get_set(py).into_any()),
            None => Self::Any(args.obj),
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
            (Self::Union | Self::Intersection | Self::Difference, Args::None)
            | (Self::Union | Self::Intersection, Args::Slf) => set.copy(),
            (Self::Difference | Self::SymmetricDifference, Args::Slf)
            | (Self::SymmetricDifference, Args::None) => PySet::empty(set.py()),
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
            (Self::Union | Self::Intersection | Self::Difference, Args::None)
            | (Self::Union | Self::Intersection, Args::Slf) => Ok(()),
            (Self::SymmetricDifference, Args::None)
            | (Self::Difference | Self::SymmetricDifference, Args::Slf) => {
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
    T: ListsMethods,
{
    match set.contains(&value) {
        Ok(true) => Ok(()),
        Ok(false) => {
            set.add(&value)?;
            list.append(value)
        }
        Err(e) => Err(e),
    }
}
