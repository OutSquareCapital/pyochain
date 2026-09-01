use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::{
            dict::{SortedDict, SortedKeyDict},
            iter::{SortedIter, SortedIterKey},
            keyset::SortedKeySet,
            set::SortedSet,
            views::BaseSortedView,
        },
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
    types::{PyBool, PyDict, PyList, PyMapping, PyNotImplemented, PySet, PyTuple, PyType},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::{py_abc, try_cast, try_cast_into};
use sorted_rs::{
    Bounds, Dir, IntOrSlice, KeysListsData, ListDataGetters, ListDataIter, ListsData,
    ListsDataMethods, SeqOrAny,
};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};
use tap::prelude::*;

pub(crate) type Reduced<'py> = PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)>;
pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>>;

#[pyclass(frozen, generic)]
pub(super) struct PyIdentity;
#[pymethods]
impl PyIdentity {
    #[staticmethod]
    fn __call__(value: Bound<'_, PyAny>) -> Bound<'_, PyAny> {
        value
    }
}
#[py_abc(
    SortedList,
    SortedKeyList,
    SortedSet,
    SortedKeySet,
    SortedDict,
    SortedKeyDict
)]
pub(super) trait SortedCollection:
    Sized + ListGetter + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool>;
    fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
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
            .get_data()
            .irange_specs(py, minimum, maximum, inclusive)?;
        self.bounded_iter(py, bounds, reverse)
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        &self,
        py: Python<'py>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let bounds = self.get_data().get_islice_specs(py, start, stop)?;
        self.bounded_iter(py, bounds, reverse)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.wrap_iter(py, ListDataIter::full(self.get_list().clone(), Dir::Fwd))
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.wrap_iter(py, ListDataIter::full(self.get_list().clone(), Dir::Bwd))
    }
    #[skip]
    fn bounded_iter<'py>(
        &self,
        py: Python<'py>,
        bounds: Option<Bounds>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match bounds {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => {
                let direction = if reverse { Dir::Bwd } else { Dir::Fwd };
                self.wrap_iter(
                    py,
                    ListDataIter::new(self.get_list().clone(), bounds, direction),
                )
            }
        }
    }

    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()>;
    fn clear(&self, py: Python<'_>);
}

#[py_abc(SortedList, SortedKeyList, SortedSet, SortedKeySet)]
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
    fn get_list(&self) -> &Arc<Mutex<Self::T>>;
    #[inline(always)]
    fn get_data(&self) -> MutexGuard<'_, Self::T> {
        match self.get_list().try_lock() {
            Ok(guard) => guard,
            //Recover if the guard was poisoned by an earlier panic instead of cascading.
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => panic!("data already locked - reentrant bug"),
        }
    }
    fn wrap_iter<'py>(
        &self,
        py: Python<'py>,
        inner: ListDataIter<Self::T>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
}
macro_rules! impl_list_getter {
    ($t:ty, $l:ty, $iter:ty) => {
        impl ListGetter for $t {
            type T = $l;
            #[inline(always)]
            fn get_list(&self) -> &Arc<Mutex<Self::T>> {
                &self.0
            }
            fn wrap_iter<'py>(
                &self,
                py: Python<'py>,
                inner: ListDataIter<Self::T>,
            ) -> PyResult<Bound<'py, abc::PyoIterator>> {
                <$iter>::new(inner).into_bound(py).map(Bound::into_super)
            }
        }
    };
}
impl_list_getter!(SortedList, ListsData, SortedIter);
impl_list_getter!(SortedKeyList, KeysListsData, SortedIterKey);
impl_list_getter!(SortedSet, ListsData, SortedIter);
impl_list_getter!(SortedKeySet, KeysListsData, SortedIterKey);
impl_list_getter!(SortedDict, ListsData, SortedIter);
impl_list_getter!(SortedKeyDict, KeysListsData, SortedIterKey);

#[py_abc(SortedList, SortedKeyList)]
pub(super) trait BaseSortedList: ListGetter + BaseSortedListSet {
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;

    #[pyo3(name = "update")]
    fn py_update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.get_data().update(py, values)
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        self.get_data().pop(py, index)
    }
    fn __add__<'py>(slf: Bound<'py, Self>, other: &Bound<'py, PyAny>)
    -> PyResult<Bound<'py, Self>>;
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().eq(other)
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().ne(other)
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().lt(other)
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().gt(other)
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().le(other)
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        self.get_data().ge(other)
    }

    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        self.get_data().delitem(py, index)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.get_data();
        match index {
            Either::Right(slice) => data
                .getitem_from_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Right),
            Either::Left(index) => data.getitem_from_int(py, index).map(Either::Left),
        }
    }
    fn __len__(&self) -> usize {
        self.get_data().length()
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
        self.get_data().imul(py, num)
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
#[py_abc(SortedSet, SortedKeySet)]
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
            let mut data = self.get_data();
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
            let mut data = self.get_data();
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
        let mut data = self.get_data();
        data.clear();
        data.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.get_data();
        match index {
            Either::Right(slice) => data
                .getitem_from_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .try_into_py()
                .map(Either::Right),
            Either::Left(index) => data.getitem_from_int(py, index).map(Either::Left),
        }
    }
    fn __delitem__(&self, py: Python<'_>, index: IntOrSlice<'_>) -> PyResult<()> {
        match index {
            Either::Right(slice) => {
                let values = self
                    .get_data()
                    .getitem_from_slice(py, &slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.get_set().bind(py).difference_update((values,))?;
                self.get_data().delitem_from_slice(py, slice)?;
            }
            Either::Left(int) => {
                let value = self.get_data().getitem_from_int(py, int)?;
                self.get_set().bind(py).remove(&value)?;
                self.get_data().delitem_from_int(py, int)?;
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
        let value = self.get_data().pop(py, index)?;
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
            let mut data = slf_ref.get_data();
            data.clear();
            data.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
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
        let mut data = slf_clone.get_data();
        set.symmetric_difference_update(other)?;
        data.clear();
        data.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
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

            fn bisect_left(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
                self.get_data().bisect_left(value)
            }

            fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
                self.get_data().bisect_right(value)
            }

            fn index(
                &self,
                value: Bound<'_, PyAny>,
                start: Option<isize>,
                stop: Option<isize>,
            ) -> PyResult<isize> {
                self.get_data().index(&value, start, stop)
            }
            fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
                self.get_data().reset(py, load)
            }
            fn clear(&self, py: Python<'_>) -> () {
                self.get_set().bind(py).clear();
                self.get_data().clear()
            }
        }
        impl BaseSortedListSet for $set {
            fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(py);
                if !set.contains(&value)? {
                    set.add(&value)?;
                    self.get_data().add(py, value)?;
                }
                Ok(())
            }
            fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(value.py());
                if set.contains(&value)? {
                    set.remove(&value)?;
                    self.get_data().remove(&value)?;
                }
                Ok(())
            }

            fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
                self.get_set().bind(value.py()).remove(&value)?;
                self.get_data().remove(value)
            }
            fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
                PySet::new(py, self.get_set().bind(py).iter()).and_then(|x| self.wrap(x))
            }
        }
    };
}
impl_sorted_collection_for_set!(SortedSet, SortedList);
impl_sorted_collection_for_set!(SortedKeySet, SortedKeyList);
#[py_abc(SortedDict, SortedKeyDict)]
pub(super) trait BaseSortedDict: ListGetter + SortedCollection {
    type KView: BaseSortedView<M = Self>;
    type VView: BaseSortedView<M = Self>;
    type IView: BaseSortedView<M = Self>;
    #[getter]
    fn get_inner(&self) -> &Py<PyDict>;
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
        self.get_inner().bind(py).len()
    }

    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.get_inner().bind(key.py()).as_any().get_item(key)
    }

    fn __delitem__(&self, key: Bound<'_, PyAny>) -> PyResult<()> {
        self.get_inner().bind(key.py()).as_any().del_item(&key)?;
        self.get_data().remove(&key)
    }
    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        if !self.__contains__(&key)? {
            self.get_data().add(py, key.clone().unbind())?;
        }
        self.get_inner().bind(py).set_item(key, value)
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
    ) -> PyResult<Bound<'py, SortedDict>> {
        let py = cls.py();
        let value = value.unwrap_or_else(|| py.None().into_bound(py));
        iterable
            .try_iter()?
            .map(|key| Ok((key?, value.clone())))
            .pipe(|v| SortedDict::try_from_iter(py, v))?
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
            self.get_data().remove(&key)?;
            self.get_inner().bind(py).pop_or_err(&key).into_pyresult()
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
            let key = self.get_data().pop(py, index)?;
            let value = self.get_inner().bind(py).pop_or_err(&key).into_pyresult()?;
            Ok((key, value))
        }
    }
    #[pyo3(signature = (index = -1))]
    fn peekitem<'py>(
        &self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let key = self.get_data().getitem_from_int(py, index)?;
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
            self.get_inner().bind(py).set_item(&key, &default)?;
            self.get_data().add(py, key.unbind())?;
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
        let mut list = self.get_data();
        let inner = self.get_inner().bind(py);
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
        let mapping = owner.get_inner().clone_ref(py).into_bound(py).into_any();
        let mapping_list = owner.get_data();
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
        match self.mapping_list.getitem_from_int(self.py, index) {
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
