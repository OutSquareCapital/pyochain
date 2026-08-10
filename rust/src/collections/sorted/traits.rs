use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::{data::ListsData, errors, iter, keyset::SortedKeySet, set::SortedSet},
    },
    pyovec::PyoVec,
    traits::IntoPyochain,
};
use either::Either;
use pyo3::{
    BoundObject, PyClass,
    exceptions::{PyIndexError, PyNotImplementedError},
    prelude::*,
    types::{
        PyBool, PyList, PyNotImplemented, PySequence, PySet, PySlice, PySliceIndices, PyTuple,
        PyType,
    },
};
use pyo3_ext::{prelude::*, types::PyAbstractSet};
use pyochain_macros::{BoundFromAny, py_abc};
use std::{
    cmp::Ordering,
    sync::{Mutex, MutexGuard, TryLockError, atomic::Ordering as AtomicOrdering},
};
use tap::prelude::*;

pub const DEFAULT_LOAD_FACTOR: usize = 1000;
pub type BoolOrNotImpl<'py> = PyResult<Either<bool, Bound<'py, PyNotImplemented>>>;
pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;
pub(crate) type Reduced<'py> = PyResult<(Bound<'py, PyType>, Bound<'py, PyTuple>)>;
pub(crate) type IntOrSlice<'py> = Either<isize, Bound<'py, PySlice>>;
pub(crate) type ObjOrVec<'py> = PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>>;
pub(super) fn try_lock_recover<'a, T>(mutex: &'a Mutex<T>, msg: &str) -> MutexGuard<'a, T> {
    match mutex.try_lock() {
        Ok(guard) => guard,
        //Recover if the guard was poisoned by an earlier panic instead of cascading.
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => panic!("{msg}"),
    }
}

#[pyclass(frozen, generic)]
pub(super) struct PyIdentity;
#[pymethods]
impl PyIdentity {
    fn __call__(&self, py: Python<'_>, value: Py<PyAny>) -> Py<PyAny> {
        value.clone_ref(py)
    }
}
#[py_abc(SortedList, SortedKeyList, SortedSet, SortedKeySet)]
pub(super) trait SortedCollection:
    Sized + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py>;
    fn __contains__(&self, value: Bound<'_, PyAny>) -> PyResult<bool>;
    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        slf: Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()>;
    fn clear(&self) -> ();
}

#[py_abc(SortedList, SortedKeyList, SortedSet, SortedKeySet)]
pub(super) trait BaseSortedListSet: SortedCollection {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
}

#[py_abc(SortedList, SortedKeyList)]
pub(super) trait SortedListGetters: BaseSortedListSet {
    #[skip]
    fn get_data(&self) -> MutexGuard<'_, ListsData>;
    fn set_load(&self, load: usize);
    #[getter]
    fn get_load(&self) -> usize;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl SortedListGetters for $t {
            #[inline(always)]
            fn get_data(&self) -> MutexGuard<'_, ListsData> {
                try_lock_recover(&self.data, "data already locked - reentrant bug")
            }
            #[inline(always)]
            fn set_load(&self, load: usize) {
                self.load.store(load, AtomicOrdering::Relaxed);
            }
            #[inline(always)]
            fn get_load(&self) -> usize {
                self.load.load(AtomicOrdering::Relaxed)
            }
        }
    };
}
impl_inner_sorted_rs!(SortedList);
impl_inner_sorted_rs!(SortedKeyList);
#[py_abc(SortedList, SortedKeyList)]
pub(super) trait BaseSortedList: SortedListGetters {
    #[skip]
    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    #[skip]
    fn delete(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
        idx: usize,
    ) -> PyResult<()>;
    #[skip]
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()>;
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    #[skip]
    fn update_from_vec(&self, py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<()>;

    fn collapse_lists<'py>(&self, py: Python<'py>) -> Vec<Py<PyAny>> {
        self.get_data().collapse(py)
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let mut data = self.get_data();
        if data.len == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }

        let (pos, idx) = {
            let len_last = data.lists.last().unwrap().len() as isize;
            match index {
                0 => (0, 0),
                -1 => {
                    let pos = data.lists.len() - 1;
                    (pos, data.lists[pos].len() - 1)
                }
                _ if 0 <= index && index < data.lists[0].len() as isize => (0, index as usize),
                _ if -len_last < index && index < 0 => {
                    let pos = data.lists.len() - 1;
                    (pos, (len_last + index) as usize)
                }
                _ => {
                    let (pos, idx) = data.pos(index)?;
                    (pos, idx as usize)
                }
            }
        };
        let val = data.lists[pos][idx].clone_ref(py);
        self.delete(py, &mut data, pos, idx)?;
        Ok(val.into_bound(py))
    }
    #[skip]
    fn delitem_from_slice(&self, py: Python<'_>, slice: Bound<'_, PySlice>) -> PyResult<()> {
        let mut data = self.get_data();
        let length = data.len as isize;
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step, start.cmp(&stop)) {
            (1, Ordering::Less) if start == 0 && stop == length => {
                drop(data);
                self.clear();
                Ok(())
            }
            (1, Ordering::Less) if length <= 8 * (stop - start) => {
                let mut values = data.getitem_from_slice(py, &PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice =
                        data.getitem_from_slice(py, &PySlice::new(py, stop, length, 1))?;
                    values.extend(new_slice);
                }
                drop(data);
                self.clear();
                self.update_from_vec(py, values)?;
                Ok(())
            }
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .rev()
                .try_for_each(|idx| {
                    let (pos, idx) = data.pos(idx)?;
                    self.delete(py, &mut data, pos, idx as usize)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        let (pos, idx) = data.pos(idx)?;
                        self.delete(py, &mut data, pos, idx as usize)
                    })
            }
        }
    }
    fn delitem_from_int(&self, py: Python<'_>, index: isize) -> PyResult<()> {
        let mut data = self.get_data();
        let (pos, idx) = data.pos(index)?;
        self.delete(py, &mut data, pos, idx as usize)
    }
    /// Return an iterator that slices sorted list using two index pairs.\
    /// The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the first inclusive and the latter exclusive.\
    /// See `_pos` for details on how an index is converted to an index pair.\
    /// When `reverse` is `True`, values are yielded from the iterator in reverse order.
    #[skip]
    fn islice_iter<'py>(
        slf: Bound<'py, Self>,
        bounds: iter::IsliceBounds,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let dir = if reverse {
            iter::Dir::Bwd
        } else {
            iter::Dir::Fwd
        };
        Self::wrap_iter(py, iter::BoundedIter::new(slf.unbind(), bounds, dir))
    }

    fn __reversed__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, abc::PyoIterator>> {
        let py = slf.py();
        Self::wrap_iter(py, iter::BoundedIter::full(slf.unbind(), iter::Dir::Bwd))
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, abc::PyoIterator>> {
        let py = slf.py();
        Self::wrap_iter(py, iter::BoundedIter::full(slf.unbind(), iter::Dir::Fwd))
    }

    fn __add__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>>;
    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        let data = self.get_data();
        match other {
            Either::Left(seq) => {
                if data.len.ne(&seq.len()?) {
                    Either::Left(false).pipe(Ok)
                } else {
                    let py = seq.py();
                    data.iter()
                        .zip(seq.try_iter()?)
                        .map(|(a, b)| a.bind(py).eq(b?))
                        .find_map(|x| match x {
                            Ok(true) => None,
                            Ok(false) => Some(Ok(false)),
                            Err(e) => Some(Err(e)),
                        })
                        .unwrap_or(Ok(true))
                        .map(Either::Left)
                }
            }

            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        let data = self.get_data();
        match other {
            Either::Left(seq) => {
                if data.len.ne(&seq.len()?) {
                    Either::Left(true).pipe(Ok)
                } else {
                    let py = seq.py();
                    data.iter()
                        .zip(seq.try_iter()?)
                        .map(|(a, b)| a.bind(py).eq(b?))
                        .find_map(|x| match x {
                            Ok(true) => None,
                            Ok(false) => Some(Ok(true)),
                            Err(e) => Some(Err(e)),
                        })
                        .unwrap_or(Ok(false))
                        .map(Either::Left)
                }
            }
            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.try_iter()?) {
                    let a = alpha.bind(py);
                    let b = beta?;
                    if a.ne(&b)? {
                        return a.lt(&b).map(Either::Left);
                    }
                }

                data.len.lt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.try_iter()?) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return Either::Left(a.gt(&b)?).pipe(Ok);
                    }
                }
                data.len.gt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.try_iter()?) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.le(b).map(Either::Left);
                    }
                }

                data.len.le(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.try_iter()?) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.ge(b).map(Either::Left);
                    }
                }

                data.len.ge(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }
            Either::Right(any) => errors::not_impl(any.py()),
        }
    }

    fn __delitem__(
        &self,
        py: Python<'_>,
        index: Either<isize, Bound<'_, PySlice>>,
    ) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.delitem_from_slice(py, slice),
            Either::Left(index) => self.delitem_from_int(py, index),
        }
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        let mut data = self.get_data();
        match index {
            Either::Right(slice) => data
                .getitem_from_slice(py, &slice)?
                .iter()
                .collect_bound::<PyList>(py)?
                .into_pyochain()
                .map(Either::Right),
            Either::Left(index) => data.getitem_from_int(py, index).map(Either::Left),
        }
    }
    fn __len__(&self) -> usize {
        self.get_data().len
    }

    fn __radd__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__add__(slf, other)
    }

    fn __rmul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        self.__mul__(py, num)
    }
    #[allow(unused_variables)]
    fn __setitem__(&self, _index: Bound<'_, PyAny>, _value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``del sl[index]`` and ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    fn __iadd__(&self, other: Bound<'_, PyAny>) -> PyResult<()> {
        self.update(&other)
    }

    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        let values = self.get_data().repeat(py, num);
        self.clear();
        self.update_from_vec(py, values)
    }

    #[allow(unused_variables)]
    fn append(&self, _value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }

    #[allow(unused_variables)]
    fn extend(&self, _values: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.update(values)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    #[allow(unused_variables)]
    fn insert(&self, _index: Bound<'_, PyAny>, _value: Bound<'_, PyAny>) -> PyResult<()> {
        let msg = "use ``sl.add(value)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
    fn reverse(&self) -> PyResult<()> {
        let msg = "use ``reversed(sl)`` instead";
        Err(PyNotImplementedError::new_err(msg))
    }
}

#[derive(BoundFromAny)]
enum SortedSetCmp<'py> {
    PySet(Bound<'py, PyAbstractSet>),
    SortedSet(Bound<'py, SortedSet>),
    SortedKeySet(Bound<'py, SortedKeySet>),
    Any(Bound<'py, PyAny>),
}

#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait BaseSortedSet: BaseSortedListSet {
    type T: BaseSortedList;
    #[skip]
    fn _set(&self) -> &Py<PySet>;
    #[skip]
    fn _list(&self) -> &Self::T;

    fn _fromset<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;

    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>>;

    fn is_disjoint<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self._set().bind(other.py()).isdisjoint(other)
    }

    fn is_subset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self._set().bind(other.py()).issubset(other)
    }

    fn is_superset<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyBool>> {
        self._set().bind(other.py()).issuperset(other)
    }

    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self._list().__getitem__(py, index)
    }
    fn __delitem__<'py>(&self, py: Python<'_>, index: IntOrSlice<'py>) -> PyResult<()> {
        let mut list = self._list().get_data();
        match index {
            Either::Right(slice) => {
                let values = list
                    .getitem_from_slice(py, &slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self._set().bind(py).difference_update((values,))?;
                self._list().delitem_from_slice(py, slice)?;
            }
            Either::Left(int) => {
                let value = list.getitem_from_int(py, int)?;
                self._set().bind(py).remove(&value)?;
                self._list().delitem_from_int(py, int)?;
            }
        };
        Ok(())
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .eq(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .eq(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).eq(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __ne__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .ne(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .ne(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).ne(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __lt__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .lt(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .lt(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).lt(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __gt__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .gt(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .gt(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).gt(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __le__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .le(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .le(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).le(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __ge__<'py>(&self, py: Python<'py>, other: SortedSetCmp<'py>) -> BoolOrNotImpl<'py> {
        match other {
            SortedSetCmp::SortedSet(o) => self
                ._set()
                .bind(py)
                .ge(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::SortedKeySet(o) => self
                ._set()
                .bind(py)
                .ge(o.get()._set().bind(py))
                .map(Either::Left),
            SortedSetCmp::PySet(pyset) => self._set().bind(py).ge(pyset).map(Either::Left),
            _ => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self._set().bind(py).len()
    }

    fn __iter__(&self) -> PyResult<Bound<'_, abc::PyoIterator>> {
        self._list().iter()
    }

    fn __reversed__(&self) -> PyResult<Bound<'_, abc::PyoIterator>> {
        self._list().rev()
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        if self._set().bind(value.py()).contains(value)? {
            Ok(1)
        } else {
            Ok(0)
        }
    }

    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let value = self._list().pop(py, index)?;
        self._set().bind(py).remove(&value)?;
        Ok(value)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let diff = self._set().bind(iterables.py()).difference(iterables)?;
        self._fromset(diff)
    }
    fn __sub__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.difference(other)
    }
    #[pyo3(signature = (*iterables))]
    fn difference_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = iterables.py();
        let set = self._set().bind(py);
        let values = iterables
            .iter()
            .flat_map(|x| x.try_iter().unwrap())
            .try_collect_bound::<PySet>(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            self._list().clear();
            self._list().update(set)?;
        } else {
            for value in values {
                self.discard(value)?
            }
        }
        Ok(self)
    }
    fn __isub__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.difference_update(other)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let intersect = self._set().bind(iterables.py()).intersection(iterables)?;
        self._fromset(intersect)
    }

    fn __and__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.intersection(other)
    }
    fn __rand__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.intersection(other)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection_update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let set = self._set().bind(iterables.py());
        set.intersection_update(iterables)?;
        self._list().clear();
        self._list().update(set)?;
        Ok(self)
    }

    fn __iand__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.intersection_update(other)
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let diff = self._set().bind(other.py()).symmetric_difference(other)?;
        self._fromset(diff)
    }
    fn __xor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn __rxor__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.symmetric_difference(other)
    }
    fn symmetric_difference_update(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let set = self._set().bind(other.py());
        set.symmetric_difference_update(other)?;
        self._list().clear();
        self._list().update(set)?;
        Ok(self)
    }
    fn __ixor__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.symmetric_difference_update(other)
    }
    fn __or__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.union(other)
    }
    fn __ror__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.union(other)
    }
    #[pyo3(signature = (*iterables))]
    fn update(&self, iterables: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = iterables.py();
        let set = self._set().bind(py);
        let values = iterables
            .iter()
            .flat_map(|x| x.try_iter().unwrap())
            .try_collect_bound::<PySet>(py)?;
        if (4 * values.len()) > set.len() {
            set.update((values,))?;
            self._list().clear();
            self._list().update(set)?;
        } else {
            for value in values.iter().map(Bound::unbind) {
                self.add(py, value)?;
            }
        }
        Ok(self)
    }

    fn __ior__(&self, other: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        self.update(other)
    }
}
