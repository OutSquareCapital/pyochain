use crate::{
    abc,
    collections::sorted::{self, iter},
    core::PyoVec,
};
use either::Either;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{KeyList, List, prelude::*, types};

pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyoVec>, Bound<'py, PyAny>>>;
pub(super) type IterRes<'py> = PyResult<Bound<'py, abc::PyoIterator>>;

macro_rules! impl_getters {
    ($($from:ty => $into:ty[$list:ty]),+ $(,)?) => {
        $(
            impl From<$from> for $into {
                fn from(inner: $from) -> Self {
                    Self(RwLock::new(inner))
                }
            }
            impl AsRef<RwLock<<$into as ListGetter>::T>> for $into {
                fn as_ref(&self) -> &RwLock<<$into as ListGetter>::T> {
                    &self.0
                }
            }
            impl ListGetter for $into {
                 type T = $from;
                 type L = $list;
             }
        )+
    };
}

impl_getters! {
    types::Dict<List> => sorted::SortedDict[List],
    types::Dict<KeyList> => sorted::SortedKeyDict[KeyList],
    KeyList => sorted::SortedKeyList[KeyList],
    List => sorted::SortedList[List],
    types::Set<List> => sorted::SortedSet[List],
    types::Set<KeyList> => sorted::SortedKeySet[KeyList],
}
#[py_abc(
    sorted::SortedList,
    sorted::SortedKeyList,
    sorted::SortedSet,
    sorted::SortedKeySet,
    sorted::SortedDict,
    sorted::SortedKeyDict
)]
pub(super) trait SortedCollectionsMethods: ListGetter {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.read().repr::<Self>(py)
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, iter::SortedIter>> {
        iter::SortedIter::build(slf.py(), slf)
    }
    fn __reversed__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, iter::SortedIterRev>> {
        iter::SortedIterRev::build(slf.py(), slf)
    }
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().as_mut().bisect::<true>(value)
    }
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().as_mut().bisect::<false>(value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.write().as_mut().index(&value, (start, stop))
    }
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        slf: &Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> IterRes<'py> {
        let bounds = slf
            .get()
            .read()
            .as_ref()
            .irange((slf.py(), minimum, maximum, inclusive))
            .transpose()?;
        iter::bounded(slf, bounds, reverse)
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        slf: &Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> IterRes<'py> {
        let bounds = slf
            .get()
            .write()
            .as_mut()
            .islice((start, stop))
            .transpose()?;
        iter::bounded(slf, bounds, reverse)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.write().as_mut().reset(py, load)
    }
}

#[py_abc(sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict)]
pub(super) trait KeyedSortedCollection: SortedCollectionsMethods
where
    Self: ListGetter<L = KeyList>,
{
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        slf: &Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> IterRes<'py> {
        let bounds = slf
            .get()
            .read()
            .as_ref()
            .irange_by_key((slf.py(), min_key, max_key, inclusive))
            .transpose()?;
        iter::bounded(slf, bounds, reverse)
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().as_mut().bisect_by_key::<true>(key)
    }
    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().as_mut().bisect_by_key::<false>(key)
    }
}
