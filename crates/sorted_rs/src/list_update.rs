use std::cmp::Ordering;

use derive_more::{Constructor, From};
use itertools::Itertools;
use parking_lot::{RwLock, RwLockReadGuard};
use pyo3::{
    prelude::*,
    types::{PyIterator, PyList},
};
use pyo3_ext::{
    prelude::{CollectBoundIterator, PyIteratorExtMethods, PyListExtMethods},
    types::PyFrozenClass,
};
use tap::prelude::*;

use crate::{ops, traits::ListsMethods, types::VecPy};

pub trait PyListsDataRef: PyFrozenClass + AsRef<RwLock<Self::L>> + From<Self::L> {
    type L: ListsMethods;
    fn try_as_updatable<'py>(
        &self,
        right: &'py Bound<'py, PyAny>,
    ) -> PyResult<ListAdd<'py, Self::L>> {
        match right.cast_exact::<Self>().map(Bound::get).ok() {
            Some(list) if std::ptr::eq(list, self) => Ok(ListAdd::Identity),
            Some(list) => list.as_ref().read().pipe(ListAdd::Sorted).pipe(Ok),
            None => right.try_iter().map(ListAdd::Iterator),
        }
    }
}
#[derive(From)]
pub enum ListAdd<'py, T> {
    Identity,
    Sorted(RwLockReadGuard<'py, T>),
    Vec(VecPy),
    Iterator(Bound<'py, PyIterator>),
}
#[derive(Constructor)]
pub struct Updater<'py, T> {
    py: Python<'py>,
    list: &'py mut T,
    py_func: Option<&'py Bound<'py, PyAny>>,
}
impl<'py, T: ListsMethods> Updater<'py, T> {
    #[inline]
    pub fn merge_with(
        mut self,
        values: ListAdd<'py, T>,
        func: impl FnMut(&Py<PyAny>, &Py<PyAny>) -> Ordering,
    ) -> PyResult<()> {
        //TODO: Check `itertools::merge` to see if it can be beneficial here.
        match values {
            ListAdd::Identity => self
                .list
                .values
                .iter_owned(self.py)
                .sorted_by(func)
                .pipe(|sorted| self.strategy(sorted)),
            ListAdd::Vec(vec) => vec
                .tap_mut(|v| v.sort_by(func))
                .into_iter()
                .pipe(|sorted| self.strategy(sorted)),
            ListAdd::Sorted(sorted) => sorted
                .values
                .iter_owned(self.py)
                .sorted_by(func)
                .pipe(|sorted| self.strategy(sorted)),
            ListAdd::Iterator(it) => it
                .into_native()
                .sorted(false, self.py_func)?
                .into_iter()
                .map(Bound::unbind)
                .pipe(|sorted| self.strategy(sorted)),
        }
    }
    fn strategy(&mut self, it: impl ExactSizeIterator<Item = Py<PyAny>>) -> PyResult<()> {
        match ops::Update::new(self.list, &it) {
            ops::Update::EmptyMaxes => self.update_chunked(it),
            ops::Update::Big => self
                .list
                .values
                .iter_owned(self.py)
                .chain(it)
                .collect_bound::<PyList>(self.py)
                .and_then(|pylist| {
                    self.py_func
                        .map_or_else(|| pylist.sort(), |key| pylist.sort_by(key, false))
                        .map(|()| pylist)
                })?
                .into_iter()
                .map(Bound::unbind)
                .pipe(|pylist| self.update_chunked(pylist)),
            ops::Update::Small => it
                .map(|x| x.into_bound(self.py))
                .try_for_each(|val| self.list.append(val)),
        }
    }

    #[inline]
    fn update_chunked(&mut self, mut it: impl ExactSizeIterator<Item = Py<PyAny>>) -> PyResult<()> {
        self.list.clear(self.py);
        self.list.len = it.len();
        let load = *self.list.load;
        std::iter::from_fn(|| {
            it.by_ref()
                .take(load)
                .collect::<VecPy>()
                .pipe(|chunk| (!chunk.is_empty()).then_some(chunk))
        })
        .pipe(|chunked| self.list.finalize_update(self.py, chunked))
    }
}
