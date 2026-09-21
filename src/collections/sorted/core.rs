use crate::{
    abc,
    collections::sorted::{self, getters::ListGetter},
    core::PyoVec,
    traits::IntoInit,
};
use either::Either;
use pyo3::prelude::*;
use pyochain_macros::py_abc;
use sorted_rs::{Bounds, KeysListsData, bisect::Bisect, iter as rsiter, prelude::*};

use tap::prelude::*;
pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyoVec>, Bound<'py, PyAny>>>;

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
        self.lock().repr::<Self>(py)
    }
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect_left(value)
    }
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect_right(value)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.as_ref()
            .clone()
            .pipe(rsiter::Full::new)
            .conv::<Self::IFull>()
            .into_bound(py)
            .map(Bound::into_super)
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.as_ref()
            .clone()
            .pipe(rsiter::FullRev::new)
            .conv::<Self::IFullRev>()
            .into_bound(py)
            .map(Bound::into_super)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.lock().list_mut().index(&value, start, stop)
    }
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let bounds = self
            .lock()
            .list_mut()
            .irange_specs(py, minimum, maximum, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        &self,
        py: Python<'py>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let bounds = self.lock().get_islice_specs(py, start, stop)?;
        self.iter_bounds(py, bounds, reverse)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.lock().list_mut().reset(py, load)
    }
}

#[py_abc(sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict)]
pub(super) trait KeyedSortedCollection: SortedCollectionsMethods
where
    Self::T: ListDataOwner<List = KeysListsData>,
{
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    fn irange_key<'py>(
        &self,
        py: Python<'py>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let data = self.lock();
        let list = data.list();
        let bounds = Bounds::from_sorted(&list.1, &list.maxes, min_key, max_key, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
    fn bisect_key_left(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect(key, Bisect::bisect_left)
    }
    fn bisect_key_right(&self, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.lock().list_mut().bisect(key, Bisect::bisect_right)
    }
}
