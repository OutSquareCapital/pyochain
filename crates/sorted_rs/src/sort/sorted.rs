use crate::{
    bounds::Loc,
    sort::{
        Chunk,
        chunk::Batch,
        keys::{KeyFn, SortKey},
        strategy::KeysCheck,
    },
    traits::ListsMethods,
    types::VecPy,
};
use derive_more::{Deref, IntoIterator};
use pyo3::prelude::*;
use tap::prelude::*;
/// Pre-compute the keys for all items, then sort by those keys.\
/// Implemented once per kind of item, since each needs its own conversion before the key function is called.
pub(crate) trait SortedByKeys<'py, T>: Sized {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>>;
}

impl<'py> SortedByKeys<'py, VecPy> for VecPy {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        key.sort_vec(self)
    }
}
impl<'py, O, I: Iterator<Item = Bound<'py, O>>> SortedByKeys<'py, Bound<'py, O>> for I {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value.into_any()).map_err(Into::into))
            .pipe(SortedVec::from_iter)
    }
}
impl<'py, I: Iterator<Item = PyResult<Bound<'py, PyAny>>>>
    SortedByKeys<'py, PyResult<Bound<'py, PyAny>>> for I
{
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value?).map_err(Into::into))
            .pipe(SortedVec::from_iter)
    }
}
impl<'py, I: Iterator<Item = Py<PyAny>>> SortedByKeys<'py, Py<PyAny>> for I {
    fn sorted_by_pykey<K: KeyFn<'py>>(self, key: &K) -> PyResult<SortedVec<K::Item>> {
        self.map(|value| key.call(value.into_bound(key.py())).map_err(Into::into))
            .pipe(SortedVec::from_iter)
    }
}

pub(crate) enum ExtendKind<T> {
    Small(T),
    Medium(T),
    Big(T),
}
impl<T: SortKey> ExtendKind<SortedVec<T>> {
    #[inline]
    pub fn extend_into<L>(self, py: Python<'_>, list: &mut L) -> PyResult<()>
    where
        L: ListsMethods<Item = T>,
    {
        match self {
            Self::Big(vec) => {
                let length = vec.len();
                let load = *list.load;
                list.clear(py);
                list.len = length;
                if length == 0 {
                } else if length <= load {
                    vec.into_iter().collect::<L::Chunk>().push_into(py, list);
                } else {
                    vec.into_iter().pipe(|mut items| {
                        std::iter::repeat_with(|| items.by_ref().take(load).collect::<L::Chunk>())
                            .take(length.div_ceil(load))
                            .for_each(|chunk| chunk.push_into(py, list));
                    });
                }
                Ok(())
            }
            Self::Medium(vec) => Batch::new(vec).extend_into(py, list),
            // Inserts the items into a non-empty list one by one, like `append`,
            // but each item is only compared from the location of the previous one, and its key isn't computed again.
            Self::Small(vec) => {
                vec.into_iter().try_fold(Loc::default(), |prev, item| {
                    let loc = prev.next_sorted(list, item.key())?;
                    if loc.idx == list.keys()[loc.pos].len() {
                        list.maxes[loc.pos] = item.key().clone().unbind();
                    }
                    list.insert_at(&loc, item);
                    list.expand(py, loc.pos);
                    list.len += 1;
                    PyResult::Ok(loc)
                })?;
                Ok(())
            }
        }
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
    fn from_iter(mut items: impl Iterator<Item = PyResult<T>>) -> PyResult<Self> {
        let capacity = items.size_hint().0;
        match items.next() {
            Some(Err(e)) => Err(e),
            Some(Ok(first)) => {
                let mut vec = Vec::with_capacity(capacity);
                let mut keys = KeysCheck::new(first.key());
                keys.check(first.key());
                vec.push(first);
                items.try_for_each(|item| {
                    let item = item?;
                    keys.check(item.key());
                    vec.push(item);
                    PyResult::Ok(())
                })?;
                keys.into_strategy().sort(vec).map(Self)
            }
            None => Ok(Self(Vec::new())),
        }
    }
    /// Same as [`Self::new`], for items already collected: sorts them without moving them to a new `Vec`.
    pub(super) fn from_vec(vec: Vec<T>) -> PyResult<Self> {
        match vec.first() {
            Some(first) => {
                let mut keys = KeysCheck::new(first.key());
                for item in &vec {
                    keys.check(item.key());
                }
                keys.into_strategy().sort(vec).map(Self)
            }
            None => Ok(Self(vec)),
        }
    }
}
