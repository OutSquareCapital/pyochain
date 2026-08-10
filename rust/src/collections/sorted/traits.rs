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
    call::PyCallArgs,
    exceptions::{PyIndexError, PyNotImplementedError},
    prelude::*,
    types::{
        PyBool, PyList, PyNotImplemented, PySequence, PySet, PySlice, PySliceIndices, PyTuple,
        PyType,
    },
};
use pyo3_ext::prelude::*;
use pyochain_macros::{py_abc, try_cast};
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

#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait BaseSortedSet: BaseSortedListSet {
    type T: BaseSortedList;
    #[skip]
    fn _list(&self) -> &Self::T;
    #[skip]
    fn from_set<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>>;
    #[getter]
    fn get_set(&self) -> &Py<PySet>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;
    #[skip]
    fn from_vec<'py>(&self, py: Python<'py>, v: Vec<Py<PyAny>>) -> PyResult<Bound<'py, Self>>;

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
                self.get_set().bind(py).difference_update((values,))?;
                self._list().delitem_from_slice(py, slice)?;
            }
            Either::Left(int) => {
                let value = list.getitem_from_int(py, int)?;
                drop(list);
                self.get_set().bind(py).remove(&value)?;
                self._list().delitem_from_int(py, int)?;
            }
        };
        Ok(())
    }

    fn __eq__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .eq(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).eq(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __ne__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .ne(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).ne(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __lt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .lt(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).lt(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __gt__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .gt(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).gt(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __le__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .le(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).le(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __ge__<'py>(&self, py: Python<'py>, other: Bound<'py, PyAny>) -> BoolOrNotImpl<'py> {
        try_cast! {
            match other {
                SortedSet | SortedKeySet => self
                    .get_set()
                    .bind(py)
                    .ge(other.get().get_set().bind(py))
                    .map(Either::Left),
                PySet => self.get_set().bind(py).ge(other).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.get_set().bind(py).len()
    }

    fn __iter__(&self) -> PyResult<Bound<'_, abc::PyoIterator>> {
        // self._list().iter()
        todo!()
    }

    fn __reversed__(&self) -> PyResult<Bound<'_, abc::PyoIterator>> {
        // self._list().rev()
        todo!()
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        difference_sorted_set(self, other.py(), (other,))
    }
    fn __isub__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        difference_update_sorted_set(slf, IntoUpdate::from_any(other)).map(|_| ())
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        intersection_sorted_set(self, other.py(), (other,))
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __or__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        union_sorted_set(self, other.py(), other.try_iter()?)
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
        let value = self._list().pop(py, index)?;
        self.get_set().bind(py).remove(&value)?;
        Ok(value)
    }
    #[pyo3(signature = (*iterables))]
    fn difference<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        difference_sorted_set(self, iterables.py(), iterables)
    }

    #[pyo3(signature = (*iterables))]
    fn difference_update<'py>(
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
            slf_ref._list().clear();
            slf_ref._list().update(set)?;
        } else {
            for value in values {
                slf_ref.discard(value)?
            }
        }
        Ok(slf)
    }
    #[pyo3(signature = (*iterables))]
    fn intersection<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        intersection_sorted_set(self, iterables.py(), iterables)
    }

    #[pyo3(signature = (*iterables))]
    fn intersection_update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        intersection_update_sorted_set(slf, iterables)
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        intersection_update_sorted_set(slf, (other,)).map(|_| ())
    }
    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.get_set()
            .bind(other.py())
            .symmetric_difference(other)
            .and_then(|diff| self.from_set(diff))
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
        let slf_ref = slf.get();
        let set = slf_ref.get_set().bind(other.py());
        set.symmetric_difference_update(other)?;
        slf_ref._list().clear();
        slf_ref._list().update(set)?;
        Ok(slf)
    }
    fn __ixor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        Self::symmetric_difference_update(slf, other).map(|_| ())
    }
    #[pyo3(signature= (*iterables))]
    fn union<'py>(&self, iterables: Bound<'py, PyTuple>) -> PyResult<Bound<'py, Self>> {
        let py = iterables.py();
        union_sorted_set(
            self,
            py,
            iterables.iter().flat_map(|x| x.try_iter().unwrap()),
        )
    }
    #[pyo3(signature = (*iterables))]
    fn update<'py>(
        slf: Bound<'py, Self>,
        iterables: Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        update_sorted_set(slf.get(), slf.py(), IntoUpdate::Tuple(iterables))?;
        Ok(slf)
    }

    fn __ior__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        update_sorted_set(slf.get(), slf.py(), IntoUpdate::from_any(other))
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
pub(super) fn update_sorted_set<'py, T: BaseSortedSet>(
    slf: &T,
    py: Python<'py>,
    other: IntoUpdate<'py>,
) -> PyResult<()> {
    let set = slf.get_set().bind(py);

    let values = other.into_set(py)?;
    if (4 * values.len()) > set.len() {
        set.update((values,))?;
        slf._list().clear();
        slf._list().update(set)?;
    } else {
        for value in values.iter().map(Bound::unbind) {
            slf.add(py, value)?;
        }
    };
    Ok(())
}
fn difference_update_sorted_set<'py, T: BaseSortedSet>(
    slf: Bound<'py, T>,
    iterables: IntoUpdate<'py>,
) -> PyResult<Bound<'py, T>> {
    let slf_ref = slf.get();
    let py = slf.py();
    let set = slf_ref.get_set().bind(py);
    let values = iterables.into_set(py)?;
    if (4 * values.len()) > set.len() {
        set.difference_update((values,))?;
        slf_ref._list().clear();
        slf_ref._list().update(set)?;
    } else {
        for value in values {
            slf_ref.discard(value)?
        }
    }
    Ok(slf)
}
fn intersection_update_sorted_set<'py, T: BaseSortedSet, O: PyCallArgs<'py>>(
    slf: Bound<'py, T>,
    iterables: O,
) -> PyResult<Bound<'py, T>> {
    let slf_ref = slf.get();
    let py = slf.py();
    let set = slf_ref.get_set().bind(py);
    set.intersection_update(iterables)?;
    slf_ref._list().clear();
    slf_ref._list().update(set)?;
    Ok(slf)
}
fn difference_sorted_set<'py, T: BaseSortedSet, O: PyCallArgs<'py>>(
    slf: &T,
    py: Python<'py>,
    iterables: O,
) -> PyResult<Bound<'py, T>> {
    slf.get_set()
        .bind(py)
        .difference(iterables)
        .and_then(|diff| slf.from_set(diff))
}
fn intersection_sorted_set<'py, T: BaseSortedSet, O: PyCallArgs<'py>>(
    slf: &T,
    py: Python<'py>,
    iterables: O,
) -> PyResult<Bound<'py, T>> {
    slf.get_set()
        .bind(py)
        .intersection(iterables)
        .and_then(|intersect| slf.from_set(intersect))
}
fn union_sorted_set<'py, T: BaseSortedSet, I: IntoIterator<Item = PyResult<Bound<'py, PyAny>>>>(
    slf: &T,
    py: Python<'py>,
    iterables: I,
) -> PyResult<Bound<'py, T>> {
    slf._list()
        .get_data()
        .iter()
        .map(|x| x.clone_ref(py).pipe(Ok))
        .chain(iterables.into_iter().map(|x| x.map(Bound::unbind)))
        .collect::<PyResult<Vec<_>>>()
        .and_then(|v| slf.from_vec(py, v))
}
