use crate::{
    abc,
    collections::sorted::{self, getters::ListGetter},
    core::PyoVec,
};
use either::Either;
use pyo3::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{IrangeSpecs, KeysListsData, RangeIdxs, iter::IterKind, prelude::*};

pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyoVec>, Bound<'py, PyAny>>>;
pub(super) type IterRes<'py> = PyResult<Bound<'py, abc::PyoIterator>>;
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
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> IterRes<'py> {
        Self::build_iter(slf, IterKind::Fwd)
    }
    fn __reversed__<'py>(slf: &Bound<'py, Self>) -> IterRes<'py> {
        Self::build_iter(slf, IterKind::Bwd)
    }
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().list_mut().bisect::<true>(value)
    }
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().list_mut().bisect::<false>(value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.write()
            .list_mut()
            .index(&value, RangeIdxs::new(start, stop))
    }
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        slf: &Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> IterRes<'py> {
        slf.get()
            .read()
            .list()
            .irange_specs(IrangeSpecs::new(slf.py(), minimum, maximum, inclusive))
            .and_then(|bounds| Self::build_iter(slf, IterKind::Bounded(bounds, reverse)))
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        slf: &Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> IterRes<'py> {
        RangeIdxs::new(start, stop)
            .into_bounds(&mut slf.get().write())
            .and_then(|bounds| Self::build_iter(slf, IterKind::Bounded(bounds, reverse)))
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.write().list_mut().reset(py, load)
    }
}

#[py_abc(sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict)]
pub(super) trait KeyedSortedCollection: SortedCollectionsMethods
where
    Self::T: ListDataOwner<List = KeysListsData>,
{
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        slf: &Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> IterRes<'py> {
        IrangeSpecs::new(slf.py(), min_key, max_key, inclusive)
            .into_bounds(slf.get().read().list())
            .and_then(|bounds| Self::build_iter(slf, IterKind::Bounded(bounds, reverse)))
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().list_mut().bisect_by_key::<true>(key)
    }
    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.write().list_mut().bisect_by_key::<false>(key)
    }
}
