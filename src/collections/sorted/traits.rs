use crate::{
    abc,
    collections::sorted::{
        self,
        dict::DictItem,
        iter::{self, PySortedIter},
    },
    core::{PyoVec, iterators},
    traits::IntoInit,
};
use either::Either;
use pyo3::{
    PyClass,
    exceptions::{PyKeyError, PyNotImplementedError},
    prelude::*,
    types::{PyBool, PyDict, PyList, PyMapping, PyNotImplemented, PySet, PySlice, PyTuple, PyType},
};
use pyo3_ext::prelude::*;
use pyo3_ext::types::{FromCmp, PyCmpOut};
use pyochain_macros::{py_abc, try_cast, try_cast_into};
use sorted_rs::{
    Bounds, InnerGetter, IntoUpdate, KeysListsData, ListDataGetters, ListDataOwner, ListsData,
    ListsDataMethods, SetData, iter as rsiter,
    types::{IntOrSlice, SeqOrAny},
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
pub(super) trait SortedCollectionsMethods:
    ListGetter + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
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
        let bounds = self
            .try_lock()
            .list_mut()
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
pub(super) trait KeyedSortedCollection: SortedCollectionsMethods + ListGetter
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
        let data = self.try_lock();
        let list = data.list();
        let bounds = Bounds::from_sorted(&list.1, list.maxes(), min_key, max_key, inclusive)?;
        self.iter_bounds(py, bounds, reverse)
    }
}
pub(super) trait ListGetter:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    type T: ListDataGetters + ListDataOwner;
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
    for [sorted::SortedList, sorted::SortedDict]
);
impl_list_getter!(
    KeysListsData,
    iter::PyBoundedKey,
    iter::PyBoundedKeyRev,
    iter::PyFullKey,
    iter::PyFullKeyRev,
    for [sorted::SortedKeyList,  sorted::SortedKeyDict]
);

impl KeyedSortedCollection for sorted::SortedKeyList {}
impl KeyedSortedCollection for sorted::SortedKeySet {}
impl KeyedSortedCollection for sorted::SortedKeyDict {}

#[py_abc(sorted::SortedList, sorted::SortedKeyList)]
pub(super) trait SortedListMethods:
    ListGetter + IntoInit + From<<Self::T as ListDataOwner>::List>
{
    fn __add__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let data = self.try_lock();
        let out = match other.cast_exact::<Self>().map(Bound::get) {
            Ok(slf) if Arc::ptr_eq(self.inner(), slf.inner()) => data.list().inner().repeat(py, 2),
            Ok(list) => data
                .list()
                .inner()
                .iter()
                .chain(list.try_lock().list().inner().iter())
                .map(|x| x.clone_ref(py))
                .collect(),
            Err(_) => data
                .list()
                .inner()
                .iter()
                .map(|x| x.clone_ref(py).pipe(Ok::<Py<PyAny>, PyErr>))
                .chain(other.try_iter()?.map(|x| x?.unbind().pipe(Ok)))
                .collect::<PyResult<Vec<Py<PyAny>>>>()?,
        };
        data.list()
            .as_owned_from(py, out)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().eq(other)
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().ne(other)
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().lt(other)
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().gt(other)
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().le(other)
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.try_lock().list().inner().ge(other)
    }

    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        self.try_lock().list_mut().del_item_or_slice(py, index)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.try_lock();
        match index {
            Either::Right(slice) => data
                .list_mut()
                .inner_mut()
                .get_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Right),
            Either::Left(index) => data
                .list_mut()
                .inner_mut()
                .get_item(py, index)
                .map(Either::Left),
        }
    }
    fn __len__(&self) -> usize {
        self.try_lock().len()
    }

    fn __radd__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__add__(other)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.try_lock()
            .list()
            .repr(py, Self::type_object(py).name()?)
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
        self.extend(&other)
    }
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .list()
            .repeat(py, num)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        self.try_lock().list_mut().imul(py, num)
    }

    #[allow(unused_variables)]
    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().add(py, value)
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().count(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.try_lock()
            .list()
            .copy(py)?
            .conv::<Self>()
            .into_bound(py)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().discard(value)
    }
    fn extend(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.try_lock().list_mut().extend(py, values)
    }
    #[allow(unused_variables)]
    fn insert(&self, index: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().list_mut().pop(py, index)
    }
    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().list_mut().remove(value)
    }
    fn reverse(&self) -> PyResult<()> {
        let msg = "use ``sl.rev()`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
}
#[py_abc(sorted::SortedSet, sorted::SortedKeySet)]
pub(super) trait SortedSetMethods: ListGetter<T = SetData<Self::L>> {
    type L: ListsDataMethods;
    #[inline(always)]
    #[skip]
    fn wrap<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>>;
    #[getter]
    #[inline(always)]
    fn get_set<'py>(&self, py: Python<'py>) -> Bound<'py, PySet> {
        self.try_lock().get_set(py)
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        match self.try_lock().__getitem__(py, index)? {
            Either::Left(list) => list.try_into_py().map(Either::Right),
            Either::Right(x) => Ok(Either::Left(x)),
        }
    }
    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        self.try_lock().del_item_or_slice(py, index)
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .eq(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).eq(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __ne__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .ne(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).ne(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __lt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .lt(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).lt(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __gt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .gt(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).gt(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __le__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .le(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).le(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __ge__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        try_cast! {
            match other {
                CaseExact::sorted::SortedSet(sorted) | CaseExact::sorted::SortedKeySet(sorted) => self
                    .get_set(py)
                    .ge(sorted.get().get_set(py))
                    .map(Either::Left),
                Case::PySet(pyset) => self.get_set(py).ge(pyset).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.try_lock().__len__(py)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let diff = self.try_lock().difference(other.py(), (other,))?;
        self.wrap(diff)
    }
    fn __isub__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        slf.get()
            .try_lock()
            .difference_update(slf.py(), other.into())
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let intersect = self.try_lock().intersection(other.py(), (other,))?;
        self.wrap(intersect)
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        slf.get().try_lock().intersection_update(slf.py(), (other,))
    }

    fn __ior__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        slf.get().try_lock().update(slf.py(), other.into())
    }
    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let union = self.try_lock().union(other.py(), (other,))?;
        self.wrap(union)
    }
    fn __ror__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__or__(other)
    }
    fn __xor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __rxor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __ixor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        Self::symmetric_difference_update(slf, other).map(|_| ())
    }
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        self.try_lock().add(py, value)
    }
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().discard(&value)
    }
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let values = self.try_lock().copy(py)?;
        self.wrap(values)
    }
    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_disjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_subset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self.try_lock().is_superset(other)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.try_lock().count(value)
    }

    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.try_lock().pop(py, index)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let diff = self.try_lock().difference(iterables.py(), iterables)?;
        self.wrap(diff)
    }

    #[pyo3(name = "difference_update", signature = (*iterables))]
    fn difference_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().try_lock().py_difference_update(&iterables)?;
        Ok(slf)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let intersect = self.try_lock().intersection(iterables.py(), iterables)?;
        self.wrap(intersect)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get()
            .try_lock()
            .intersection_update(slf.py(), iterables)?;
        Ok(slf)
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.try_lock().remove(value)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let diff = self.try_lock().symmetric_difference(other)?;
        self.wrap(diff)
    }
    fn symmetric_difference_update<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().try_lock().symmetric_difference_update(other)?;
        // NOTE: the clone here is cheap (just an incref) and necessary to return `Self`
        Ok(slf.clone())
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let union = self.try_lock().union(iterables.py(), iterables)?;
        self.wrap(union)
    }
    #[pyo3(name ="update", signature = (*iterables))]
    fn py_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get()
            .try_lock()
            .update(slf.py(), IntoUpdate::Tuple(iterables))?;
        Ok(slf)
    }
}
impl<T: SortedSetMethods> SortedCollectionsMethods for T {
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.get_set(value.py()).contains(value)
    }
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        PyTuple::new(py, [self.get_set(py).clone()]).map(|tup| (Self::type_object(py), tup))
    }

    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_left(value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.try_lock().list_mut().bisect_right(value)
    }
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<usize> {
        self.try_lock().list_mut().index(&value, start, stop)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        self.try_lock().reset(py, load)
    }
    fn clear(&self, py: Python<'_>) {
        self.try_lock().clear(py);
    }
}
#[py_abc(
    sorted::SortedItemsView,
    sorted::SortedKeysView,
    sorted::SortedValuesView,
    sorted::SortedByKeyItemsView,
    sorted::SortedByKeyKeysView,
    sorted::SortedByKeyValuesView
)]
pub trait SortedViewMethods:
    PyClass<BaseType = abc::PyoSequence> + abc::traits::MappingView + Send + Sync
where
    Self::M: SortedDictMethods + PyClass,
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
                    let keys = data.list_mut().inner_mut().get_slice(py, &slice)?;
                    data.list_mut().del_slice(py, slice)?;
                    for key in keys {
                        dict.del_item(key)?;
                    }
                    Ok(())
                },
                int => {
                    let key = mapping.try_lock().list_mut().pop(py, int.extract::<isize>()?)?;
                    dict.del_item(key)?;
                    Ok(())
                }
            }
        }
    }
}

#[py_abc(sorted::SortedDict, sorted::SortedKeyDict)]
pub(super) trait SortedDictMethods:
    ListGetter + SortedCollectionsMethods + IntoInit
{
    type KView: SortedViewMethods<M = Self>;
    type VView: SortedViewMethods<M = Self>;
    type IView: SortedViewMethods<M = Self>;
    #[getter]
    fn get_dict(&self) -> &Py<PyDict>;
    #[skip]
    fn copy_from_iter<'py, I: IntoIterator<Item = PyResult<DictItem<'py>>>>(
        &self,
        py: Python<'py>,
        v: I,
    ) -> PyResult<Self>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
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
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.iter(py)
            .pipe(|v| self.copy_from_iter(py, v))?
            .into_bound(py)
    }
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
        self.try_lock().list_mut().remove(&key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        if !self.__contains__(&key)? {
            self.try_lock().list_mut().add(py, key.clone().unbind())?;
        }
        self.get_dict().bind(py).set_item(key, value)
    }
    fn __or__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = self.iter(py).chain(value.pipe(iter_mapping)?);
        self.copy_from_iter(py, items)?.into_bound(py)
    }
    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(other.py(), Some(other), None)
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __ror__<'py>(&self, value: &Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let items = value.pipe(iter_mapping)?.chain(self.iter(py));
        self.copy_from_iter(py, items)?.into_bound(py)
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
            .pipe(|v| sorted::SortedDict::empty(py).copy_from_iter(py, v))?
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
            self.try_lock().list_mut().remove(&key)?;
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
            let key = self.try_lock().list_mut().pop(py, index)?;
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
        let key = self.try_lock().list_mut().inner_mut().get_item(py, index)?;
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
            self.try_lock().list_mut().add(py, key.unbind())?;
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
                .pipe(|v| list.list_mut().extend(py, v))?;
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
                list.list_mut().clear();
                inner
                    .iter()
                    .map(|(k, _)| k.unbind())
                    .collect::<Vec<_>>()
                    .pipe(|v| list.list_mut().extend(py, v))?;
                Ok(())
            } else {
                pairs.keys_view().iter_py().try_for_each(|key| {
                    let k = key?;
                    let new = pairs.as_any().get_item(&k)?;
                    self.__setitem__(k, new)
                })
            }
        }
    }
}

pub(super) struct SortedDictIter<'a, 'py, D: SortedDictMethods> {
    py: Python<'py>,
    mapping: Bound<'py, PyAny>,
    mapping_list: MutexGuard<'a, D::T>,
    range: std::ops::Range<isize>,
}
impl<'a, 'py, D: SortedDictMethods> SortedDictIter<'a, 'py, D> {
    fn new(owner: &'a D, py: Python<'py>) -> Self {
        let mapping = owner.get_dict().clone_ref(py).into_bound(py).into_any();
        let mapping_list = owner.try_lock();
        let range = 0..mapping_list.len().cast_signed();
        Self {
            py,
            mapping,
            mapping_list,
            range,
        }
    }
}
impl<'py, D: SortedDictMethods> Iterator for SortedDictIter<'_, 'py, D> {
    type Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>;
    fn next(&mut self) -> Option<PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let index = self.range.next()?;
        // NOTE: I tried to avoid double match here, but the `get_item` error caused reference issues.
        match self
            .mapping_list
            .list_mut()
            .inner_mut()
            .get_item(self.py, index)
        {
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
fn iter_mapping<'py>(
    mapping: &Bound<'py, PyMapping>,
) -> PyResult<impl Iterator<Item = PyResult<DictItem<'py>>>> {
    mapping
        .call_method0("items")?
        .try_iter()?
        .map(|iter| iter?.extract::<DictItem>())
        .pipe(Ok)
}
