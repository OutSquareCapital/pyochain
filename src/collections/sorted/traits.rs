use crate::{
    abc,
    collections::sorted::{
        self,
        iter::{self, PySortedIter},
    },
    core::{PyoVec, iterators},
    traits::IntoInit,
};
use either::Either;
use pyo3::{
    PyClass, PyTypeInfo,
    call::PyCallArgs,
    exceptions::{PyKeyError, PyNotImplementedError},
    prelude::*,
    types::{PyBool, PyDict, PyList, PyMapping, PyNotImplemented, PySet, PySlice, PyTuple, PyType},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::{py_abc, try_cast, try_cast_into};
use sorted_rs::{
    Bounds, InnerGetter, IntOrSlice, KeysListsData, ListDataGetters, ListsData, ListsDataMethods,
    SeqOrAny, iter as rsiter,
};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use tap::prelude::*;

pub(crate) type Reduced<'py> = PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)>;
pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>>;

#[py_abc(
    sorted::SortedList,
    sorted::SortedKeyList,
    sorted::SortedSet,
    sorted::SortedKeySet,
    sorted::SortedDict,
    sorted::SortedKeyDict
)]
pub(super) trait SortedCollection:
    Sized + ListGetter + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool>;
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize>;
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
            .try_lock()
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
        let bounds = self
            .try_lock()
            .inner_mut()
            .get_islice_specs(py, start, stop)?;
        self.iter_bounds(py, bounds, reverse)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.inner()
            .clone()
            .pipe(rsiter::Full::new)
            .conv::<Self::IFull>()
            .into_pyiterator(py)
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.inner()
            .clone()
            .pipe(rsiter::FullRev::new)
            .conv::<Self::IFullRev>()
            .into_pyiterator(py)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize>;
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()>;
    fn clear(&self, py: Python<'_>);
}

#[py_abc(sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict)]
pub(super) trait KeyedSortedCollection:
    SortedCollection + ListGetter<T = KeysListsData>
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
        let data = self.try_lock();
        let bounds = Bounds::from_sorted(&data.1, data.maxes(), min_key, max_key, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
}

#[py_abc(
    sorted::SortedList,
    sorted::SortedKeyList,
    sorted::SortedSet,
    sorted::SortedKeySet
)]
pub(super) trait BaseSortedListSet: SortedCollection {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()>;
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
}
pub(super) trait ListGetter:
    Sized + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    type T: ListDataGetters + ListsDataMethods;
    type I: PySortedIter + From<rsiter::Bounded<Self::T>>;
    type IRev: PySortedIter + From<rsiter::BoundedRev<Self::T>>;
    type IFull: PySortedIter + From<rsiter::Full<Self::T>>;
    type IFullRev: PySortedIter + From<rsiter::FullRev<Self::T>>;
    fn inner(&self) -> &Arc<Mutex<Self::T>>;
    #[inline(always)]
    fn try_lock(&self) -> MutexGuard<'_, Self::T> {
        match self.inner().try_lock() {
            Ok(guard) => guard,
            //Recover if the guard was poisoned by an earlier panic instead of cascading.
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("data already locked - reentrant bug"),
        }
    }
    fn iter_bounds<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match (bounds, reverse) {
            (None, _) => iterators::Iter::empty(py).map(Bound::into_super),
            (Some(bounds), true) => rsiter::BoundedRev::new(self.inner().clone(), bounds)
                .conv::<Self::IRev>()
                .into_pyiterator(py),
            (Some(bounds), false) => rsiter::Bounded::new(self.inner().clone(), bounds)
                .conv::<Self::I>()
                .into_pyiterator(py),
        }
    }
}
macro_rules! impl_list_getter {
    ($l:ty, $iter:ty, $iter_rev:ty, $iter_full:ty, $iter_full_rev:ty, for [$($t:ty),+ $(,)?]) => {
        $(
            impl ListGetter for $t {
                type T = $l;
                type I = $iter;
                type IRev = $iter_rev;
                type IFull = $iter_full;
                type IFullRev = $iter_full_rev;
                #[inline(always)]
                fn inner(&self) -> &Arc<Mutex<Self::T>> {
                    &self.0
                }
            }
        )+
    };
}
impl_list_getter!(
    ListsData,
    iter::PyBounded,
    iter::PyBoundedRev,
    iter::PyFull,
    iter::PyFullRev,
    for [sorted::SortedList, sorted::SortedSet, sorted::SortedDict]
);
impl_list_getter!(
    KeysListsData,
    iter::PyBoundedKey,
    iter::PyBoundedKeyRev,
    iter::PyFullKey,
    iter::PyFullKeyRev,
    for [sorted::SortedKeyList, sorted::SortedKeySet, sorted::SortedKeyDict]
);

impl KeyedSortedCollection for sorted::SortedKeyList {}
impl KeyedSortedCollection for sorted::SortedKeySet {}
impl KeyedSortedCollection for sorted::SortedKeyDict {}

#[py_abc(sorted::SortedList, sorted::SortedKeyList)]
pub(super) trait BaseSortedList: ListGetter + BaseSortedListSet {
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;

    #[pyo3(name = "update")]
    fn py_update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.try_lock().update(py, values)
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().pop(py, index)
    }
    fn __add__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>)
    -> PyResult<Bound<'py, Self>>;
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().eq(other)
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().ne(other)
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().lt(other)
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().gt(other)
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().le(other)
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().inner().ge(other)
    }

    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        self.try_lock().del_item_or_slice(py, index)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.try_lock();
        match index {
            Either::Right(slice) => data
                .inner_mut()
                .get_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Right),
            Either::Left(index) => data.inner_mut().get_item(py, index).map(Either::Left),
        }
    }
    fn __len__(&self) -> usize {
        self.try_lock().length()
    }

    fn __radd__<'py>(
        slf: Bound<'py, Self>,
        other: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__add__(slf, other)
    }

    fn __rmul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.__mul__(py, num)
    }
    #[allow(unused_variables)]
    fn __setitem__(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``del sl[index]`` and ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    fn __iadd__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.py_update(&other)
    }

    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        self.try_lock().imul(py, num)
    }

    #[allow(unused_variables)]
    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    #[allow(unused_variables)]
    fn extend(&self, values: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.update(values)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    #[allow(unused_variables)]
    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    fn reverse(&self) -> PyResult<()> {
        let msg = "use ``reversed(sl)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
}
#[py_abc(sorted::SortedSet, sorted::SortedKeySet)]
pub(super) trait BaseSortedSet: ListGetter + BaseSortedListSet {
    #[inline(always)]
    #[skip]
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>>;
    #[getter]
    fn get_set(&self) -> &Py<PySet>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;

    #[skip]
    fn update<'py>(&self, py: Python<'py>, other: IntoUpdate<'py>) -> PyResult<()> {
        let set = self.get_set().bind(py);

        let values = other.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.update((values,))?;
            let mut data = self.try_lock();
            data.clear();
            data.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        } else {
            for value in values.iter().map(Bound::unbind) {
                self.add(py, value)?;
            }
        }
        Ok(())
    }
    #[skip]
    fn difference<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, Self>> {
        self.get_set()
            .bind(py)
            .difference(iterables)
            .and_then(|diff| self.wrap(diff))
    }
    #[skip]
    fn intersection<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, Self>> {
        self.get_set()
            .bind(py)
            .intersection(iterables)
            .and_then(|intersect| self.wrap(intersect))
    }
    #[skip]
    fn union<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<Bound<'py, Self>> {
        self.get_set()
            .bind(py)
            .union(iterables)
            .and_then(|u| self.wrap(u))
    }
    #[skip]
    fn difference_update(&self, py: Python<'_>, iterables: IntoUpdate<'_>) -> PyResult<()> {
        let set = self.get_set().bind(py);
        let values = iterables.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            let mut data = self.try_lock();
            data.clear();
            data.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        } else {
            for value in values {
                self.discard(value)?;
            }
        }
        Ok(())
    }
    #[skip]
    fn intersection_update<'py, O: PyCallArgs<'py>>(
        &self,
        py: Python<'py>,
        iterables: O,
    ) -> PyResult<()> {
        let set = self.get_set().bind(py);
        set.intersection_update(iterables)?;
        let mut data = self.try_lock();
        data.clear();
        data.update(py, set.iter().map(Bound::unbind).collect())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.try_lock();
        match index {
            Either::Right(slice) => data
                .inner_mut()
                .get_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Right),
            Either::Left(index) => data.inner_mut().get_item(py, index).map(Either::Left),
        }
    }
    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => {
                let values = self
                    .try_lock()
                    .inner_mut()
                    .get_slice(py, &slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.get_set().bind(py).difference_update((values,))?;
                self.try_lock().del_slice(py, slice)?;
            }
            Either::Left(int) => {
                let value = self.try_lock().inner_mut().get_item(py, int)?;
                self.get_set().bind(py).remove(&value)?;
                self.try_lock().del_item(py, int)?;
            }
        }
        Ok(())
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .eq(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).eq(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __ne__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .ne(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).ne(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __lt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .lt(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).lt(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __gt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .gt(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).gt(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __le__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .le(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).le(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __ge__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::SortedSet(sorted) | CaseExact::SortedKeySet(sorted) => self
                    .get_set()
                    .bind(py)
                    .ge(sorted.get().get_set().bind(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set().bind(py).ge(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.get_set().bind(py).len()
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.difference(other.py(), (other,))
    }
    fn __isub__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        slf.get()
            .difference_update(slf.py(), IntoUpdate::from_any(other))?;
        Ok(())
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.intersection(other.py(), (other,))
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        slf.get().intersection_update(slf.py(), (other,))
    }

    fn __ior__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        slf.get().update(slf.py(), IntoUpdate::from_any(other))
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.union(other.py(), (other,))
    }
    fn __ror__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(other)
    }

    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.get_set().bind(other.py()).isdisjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.get_set().bind(other.py()).issubset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.get_set().bind(other.py()).issuperset(other)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        if self.get_set().bind(value.py()).contains(value)? {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let value = self.try_lock().pop(py, index)?;
        self.get_set().bind(py).remove(&value)?;
        Ok(value)
    }
    #[pyo3(name ="difference", signature = (*iterables))]
    fn py_difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.difference(iterables.py(), iterables)
    }

    #[pyo3(name = "difference_update", signature = (*iterables))]
    fn py_difference_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        let slf_ref = slf.get();
        let py = iterables.py();
        let set = slf_ref.get_set().bind(py);
        let values = iterables
            .iter()
            .flat_map(|x| x.try_iter().unwrap())
            .try_collect_bound::<PySet>(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            let mut data = slf_ref.try_lock();
            data.clear();
            data.update(py, set.iter().map(Bound::unbind).collect())?;
        } else {
            for value in values {
                slf_ref.discard(value)?;
            }
        }
        Ok(slf)
    }
    #[pyo3(name= "intersection", signature = (*iterables))]
    fn py_intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.intersection(iterables.py(), iterables)
    }

    #[pyo3(name = "intersection_update", signature = (*iterables))]
    fn py_intersection_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get()
            .intersection_update(slf.py(), iterables)
            .map(|()| slf)
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.get_set()
            .bind(other.py())
            .symmetric_difference(other)
            .and_then(|diff| self.wrap(diff))
    }
    fn __xor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __rxor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn symmetric_difference_update<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let slf_clone = slf.get();
        let set = slf_clone.get_set().bind(other.py());
        let mut data = slf_clone.try_lock();
        set.symmetric_difference_update(other)?;
        data.clear();
        data.update(py, set.iter().map(Bound::unbind).collect())?;
        // NOTE: the clone here is cheap (just an incref) and necessary to return `Self`
        Ok(slf.clone())
    }
    fn __ixor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        Self::symmetric_difference_update(slf, other).map(|_| ())
    }
    #[pyo3(name = "union", signature= (*iterables))]
    fn py_union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        self.union(iterables.py(), iterables)
    }
    #[pyo3(name ="update", signature = (*iterables))]
    fn py_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get()
            .update(slf.py(), IntoUpdate::Tuple(iterables))
            .map(|()| slf)
    }
}
pub(super) enum IntoUpdate<'py> {
    Set(Bound<'py, PySet>),
    Tuple(Bound<'py, PyTuple>),
    Any(Bound<'py, PyAny>),
}
impl<'py> IntoUpdate<'py> {
    pub(super) fn from_any(other: Bound<'py, PyAny>) -> Self {
        if other.is_exact_instance_of::<PySet>() {
            Self::Set(unsafe { other.cast_into_unchecked::<PySet>() })
        } else {
            Self::Any(other)
        }
    }
    fn into_set(self, py: Python<'py>) -> PyResult<Bound<'py, PySet>> {
        match self {
            IntoUpdate::Tuple(tup) => tup
                .iter()
                .flat_map(|x| x.try_iter().unwrap())
                .try_collect_bound::<PySet>(py),
            IntoUpdate::Set(pyset) => Ok(pyset),
            IntoUpdate::Any(any) => any.try_iter()?.try_collect_bound::<PySet>(py),
        }
    }
}

macro_rules! impl_sorted_collection_for_set {
    ($set:ty, $list:ty) => {
        impl SortedCollection for $set {
            fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
                self.get_set().bind(value.py()).contains(value)
            }
            fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
                PyTuple::new(py, [self.get_set().clone_ref(py)])
                    .map(|tup| (Self::type_object(py), tup))
            }

            fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
                self.try_lock().bisect_left(value)
            }

            fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
                self.try_lock().bisect_right(value)
            }

            fn index(
                &self,
                value: Bound<'_, PyAny>,
                start: Option<isize>,
                stop: Option<isize>,
            ) -> PyResult<usize> {
                self.try_lock().index(&value, start, stop)
            }
            fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
                self.try_lock().reset(py, load)
            }
            fn clear(&self, py: Python<'_>) -> () {
                self.get_set().bind(py).clear();
                self.try_lock().clear()
            }
        }
        impl BaseSortedListSet for $set {
            fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(py);
                if !set.contains(&value)? {
                    set.add(&value)?;
                    self.try_lock().add(py, value)?;
                }
                Ok(())
            }
            fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(value.py());
                if set.contains(&value)? {
                    set.remove(&value)?;
                    self.try_lock().remove(&value)?;
                }
                Ok(())
            }

            fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
                self.get_set().bind(value.py()).remove(&value)?;
                self.try_lock().remove(value)
            }
            fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
                PySet::new(py, self.get_set().bind(py).iter()).and_then(|x| self.wrap(x))
            }
        }
    };
}
impl_sorted_collection_for_set!(sorted::SortedSet, sorted::SortedList);
impl_sorted_collection_for_set!(sorted::SortedKeySet, sorted::SortedKeyList);

#[py_abc(
    sorted::SortedItemsView,
    sorted::SortedKeysView,
    sorted::SortedValuesView,
    sorted::SortedByKeyItemsView,
    sorted::SortedByKeyKeysView,
    sorted::SortedByKeyValuesView
)]
pub trait BaseSortedView:
    Sized + PyClass<BaseType = abc::PyoSequence> + abc::traits::MappingView + Send + Sync
where
    Self::M: BaseSortedDict + PyClass,
{
    #[skip]
    fn new(mapping: Bound<'_, Self::M>) -> Self;
    fn __getitem__<'py>(&self, index: Bound<'py, PyAny>) -> ObjOrVec<'py>;
    fn __delitem__(&self, index: Bound<'_, PyAny>) -> PyResult<()> {
        let py = index.py();
        let mapping = self.mapping().get();
        let dict = mapping.get_dict().bind(py);
        try_cast_into! {
            match index {
                Case::PySlice(slice) => {
                    let mut data = mapping.try_lock();
                    let keys = data.inner_mut().get_slice(py, &slice)?;
                    data.del_slice(py, slice)?;
                    for key in keys {
                        dict.del_item(key)?;
                    }
                    Ok(())
                },
                int => {
                    let key = mapping.try_lock().pop(py, int.extract::<isize>()?)?;
                    dict.del_item(key)?;
                    Ok(())
                }
            }
        }
    }
}

#[py_abc(sorted::SortedDict, sorted::SortedKeyDict)]
pub(super) trait BaseSortedDict: ListGetter + SortedCollection {
    type KView: BaseSortedView<M = Self>;
    type VView: BaseSortedView<M = Self>;
    type IView: BaseSortedView<M = Self>;
    #[getter]
    fn get_dict(&self) -> &Py<PyDict>;
    fn keys(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::KView>> {
        let py = slf.py();
        Self::KView::new(slf).into_bound(py)
    }
    fn items(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::IView>> {
        let py = slf.py();
        Self::IView::new(slf).into_bound(py)
    }
    fn values(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self::VView>> {
        let py = slf.py();
        Self::VView::new(slf).into_bound(py)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>>;
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
    #[skip]
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.__contains__(value)
    }
    #[skip]
    fn len(&self, py: Python<'_>) -> usize {
        self.__len__(py)
    }
    #[skip]
    fn iter<'py>(&self, py: Python<'py>) -> SortedDictIter<'_, 'py, Self> {
        SortedDictIter::new(self, py)
    }
    fn __len__(&self, py: Python<'_>) -> usize {
        self.get_dict().bind(py).len()
    }

    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.get_dict().bind(key.py()).as_any().get_item(key)
    }

    fn __delitem__(&self, key: Bound<'_, PyAny>) -> PyResult<()> {
        self.get_dict().bind(key.py()).as_any().del_item(&key)?;
        self.try_lock().remove(&key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        if !self.__contains__(&key)? {
            self.try_lock().add(py, key.clone().unbind())?;
        }
        self.get_dict().bind(py).set_item(key, value)
    }

    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other.py(), Some(other), None)
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }

    #[classmethod]
    #[pyo3(signature = (iterable, value = None, /))]
    fn from_keys<'py>(
        cls: Bound<'py, PyType>,
        iterable: Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, sorted::SortedDict>> {
        let py = cls.py();
        let value = value.unwrap_or_else(|| py.None().into_bound(py));
        iterable
            .try_iter()?
            .map(|key| Ok((key?, value.clone())))
            .pipe(|v| sorted::SortedDict::try_from_iter(py, v))?
            .into_bound(py)
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        if self.__contains__(&key)? {
            self.try_lock().remove(&key)?;
            self.get_dict().bind(py).pop_or_err(&key).into_pyresult()
        } else {
            default.ok_or_else(|| PyKeyError::new_err(key.to_string()))
        }
    }

    #[pyo3(signature = (index = -1))]
    fn popitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if self.len(py) == 0 {
            let msg = "popitem(): dictionary is empty";
            Err(PyKeyError::new_err(msg))
        } else {
            let key = self.try_lock().pop(py, index)?;
            let value = self.get_dict().bind(py).pop_or_err(&key).into_pyresult()?;
            Ok((key, value))
        }
    }
    #[pyo3(signature = (index = -1))]
    fn peekitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let key = self.try_lock().inner_mut().get_item(py, index)?;
        self.__getitem__(&key).map(|value| (key, value))
    }
    #[pyo3(signature = (key, default = None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let py = key.py();
        if self.__contains__(&key)? {
            self.__getitem__(&key).map(Some)
        } else {
            self.get_dict().bind(py).set_item(&key, &default)?;
            self.try_lock().add(py, key.unbind())?;
            Ok(default)
        }
    }
    #[pyo3(signature = (m = None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let mut list = self.try_lock();
        let inner = self.get_dict().bind(py);
        if self.len(py) == 0 {
            if let Some(it) = m {
                try_cast! {
                    match it {
                        CaseExact::PyDict(d) => inner.update(d.as_mapping())?,
                        Case::PyMapping(m) => inner.update(m)?,
                        iterable => inner.update_from_sequence(&iterable)?,
                    }
                }
            }
            if let Some(kw) = kwargs {
                inner.update(kw.as_mapping())?;
            }

            inner
                .iter()
                .map(|(k, _)| k.unbind())
                .collect::<Vec<_>>()
                .pipe(|v| list.update(py, v))?;
            Ok(())
        } else {
            let pairs = try_cast_into! {match (m, kwargs) {
                (Some(CaseExact::PyDict(d)), None) => d,
                (Some(CaseExact::PyDict(d)), Some(kw)) => {
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(Case::PyMapping(m)), None) => {
                    let d = PyDict::new(py);
                    d.update(&m)?;
                    d
                }
                (Some(Case::PyMapping(m)), Some(kw)) => {
                    let d = PyDict::new(py);
                    d.update(&m)?;
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(iterable), Some(kw)) => {
                    let d = PyDict::from_sequence(&iterable)?;
                    d.update(kw.as_mapping())?;
                    d
                }
                (Some(iterable), None) => PyDict::from_sequence(&iterable)?,
                (None, Some(kw)) => kw,
                (None, None) => PyDict::new(py),
            }};
            if (10 * pairs.len()) > self.len(py) {
                inner.update(pairs.as_mapping())?;
                list.clear();
                inner
                    .iter()
                    .map(|(k, _)| k.unbind())
                    .collect::<Vec<_>>()
                    .pipe(|v| list.update(py, v))?;
                Ok(())
            } else {
                for key in pairs.keys_view().iter_py() {
                    let k = key?;
                    let new = pairs.as_any().get_item(&k)?;
                    self.__setitem__(k, new)?;
                }
                Ok(())
            }
        }
    }
}

pub(super) struct SortedDictIter<'a, 'py, D: BaseSortedDict> {
    py: Python<'py>,
    mapping: Bound<'py, PyAny>,
    mapping_list: MutexGuard<'a, D::T>,
    range: std::ops::Range<isize>,
}
impl<'a, 'py, D: BaseSortedDict> SortedDictIter<'a, 'py, D> {
    fn new(owner: &'a D, py: Python<'py>) -> Self {
        let mapping = owner.get_dict().clone_ref(py).into_bound(py).into_any();
        let mapping_list = owner.try_lock();
        let range = 0..mapping_list.length().cast_signed();
        Self {
            py,
            mapping,
            mapping_list,
            range,
        }
    }
}
impl<'py, D: BaseSortedDict> Iterator for SortedDictIter<'_, 'py, D> {
    type Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>;
    fn next(&mut self) -> Option<PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let index = self.range.next()?;
        // NOTE: I tried to avoid double match here, but the `get_item` error caused reference issues.
        match self.mapping_list.inner_mut().get_item(self.py, index) {
            Ok(key) => {
                let value = self.mapping.get_item(&key);
                match value {
                    Ok(v) => Some(Ok((key, v))),
                    Err(e) => Some(Err(e)),
                }
            }
            Err(e) => Some(Err(e)),
        }
    }
}
