use crate::{
    sort::{
        Chunk,
        keys::{KeyFn, SortKey},
        strategy::{Element, SortBy},
    },
    traits::ListsMethods,
};
use derive_more::{Deref, IntoIterator};
use pyo3::prelude::*;
use tap::Pipe;

/// Pre-compute the keys for all items, then sort by those keys.\
/// Implemented once per kind of item, since each needs its own conversion before the key function is called.
pub(crate) trait SortedByKeys<'py, T>: Iterator<Item = T> + Sized {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>>;
}

impl<'py, O, I: Iterator<Item = Bound<'py, O>>> SortedByKeys<'py, Bound<'py, O>> for I {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value.into_any()).map_err(Into::into))
            .pipe(SortedVec::new)
    }
}
impl<'py, I: Iterator<Item = PyResult<Bound<'py, PyAny>>>>
    SortedByKeys<'py, PyResult<Bound<'py, PyAny>>> for I
{
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value?).map_err(Into::into))
            .pipe(SortedVec::new)
    }
}
impl<'py, I: Iterator<Item = Py<PyAny>>> SortedByKeys<'py, Py<PyAny>> for I {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value.into_bound(key.py())).map_err(Into::into))
            .pipe(SortedVec::new)
    }
}

/// A sorted vec, ready to be added to a sorted list.
#[derive(Deref, IntoIterator)]
pub struct SortedVec<T>(Vec<T>);
impl<T: SortKey> SortedVec<T> {
    /// Citation from `CPython`'s listobject.c for the first phase of the sort:\
    /// > The pre-sort check: here's where we decide which compare function to use.\
    /// > How much optimization is safe?\
    /// > We test for homogeneity with respect to  several properties that are expensive to check at compare-time.
    fn new(mut items: impl Iterator<Item = PyResult<T>>) -> PyResult<Self> {
        let capacity = items.size_hint().0;
        match items.next() {
            Some(Err(e)) => Err(e),
            Some(Ok(first)) => {
                let mut vec = Vec::with_capacity(capacity);
                let mut strat = Some(SortBy::check_first(&first));
                vec.push(first);
                items.try_for_each(|item| {
                    let item = item?;
                    strat = strat.and_then(|state| state.check_next(item.key()));
                    vec.push(item);
                    PyResult::Ok(())
                })?;
                strat
                    .unwrap_or(SortBy::Element(Element::SAFE))
                    .sort(vec)
                    .map(Self)
            }
            None => Ok(Self(Vec::new())),
        }
    }
}
pub trait ExtendList<T: SortKey>: IntoIterator<Item = T> + Sized {
    fn len(&self) -> usize;
    #[inline]
    fn extend_into<L>(self, py: Python<'_>, list: &mut L)
    where
        L: ListsMethods<Item = T>,
    {
        let length = self.len();
        let load = *list.load;
        list.clear(py);
        list.len = length;
        if length == 0 {
        } else if length <= load {
            self.into_iter().collect::<L::Chunk>().push_into(py, list);
        } else {
            self.into_iter().pipe(|mut items| {
                std::iter::repeat_with(|| items.by_ref().take(load).collect::<L::Chunk>())
                    .take(length.div_ceil(load))
                    .for_each(|chunk| chunk.push_into(py, list));
            });
        }
    }
}
impl<T: SortKey> ExtendList<T> for SortedVec<T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}
