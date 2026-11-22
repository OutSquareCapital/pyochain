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
        self.map(|value| key.call(value.into_any()))
            .collect::<Result<Vec<_>, K::Error>>()
            .map_err(Into::into)
            .and_then(SortedVec::new)
    }
}
impl<'py, I: Iterator<Item = PyResult<Bound<'py, PyAny>>>>
    SortedByKeys<'py, PyResult<Bound<'py, PyAny>>> for I
{
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value?).map_err(Into::into))
            .collect::<PyResult<Vec<_>>>()
            .and_then(SortedVec::new)
    }
}
impl<'py, I: Iterator<Item = Py<PyAny>>> SortedByKeys<'py, Py<PyAny>> for I {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value.into_bound(key.py())))
            .collect::<Result<Vec<_>, K::Error>>()
            .map_err(Into::into)
            .and_then(SortedVec::new)
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
    fn new(items: Vec<T>) -> PyResult<Self> {
        match items.split_first() {
            Some((first, rest)) => {
                rest.iter()
                    .map(SortKey::key)
                    // Stops with `None` once the safe compare is the only option left:
                    // - in a list of non-tuples, once the state has fallen back to `Safe`;
                    // - in a list of tuples, when an item isn't a non-empty tuple.
                    .try_fold(SortBy::check_first(first), SortBy::check_next)
                    .unwrap_or(SortBy::Element(Element::SAFE))
                    .sort(items)
                    .map(Self)
            }

            None => Ok(Self(items)),
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
