use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::{data::ListsData, errors, iter},
    },
    iterators,
    pyo3_ext::prelude::*,
    pyovec::PyoVec,
    traits::IntoPyochain,
};
use either::Either;
use pyo3::{
    PyClass,
    exceptions::{PyIndexError, PyNotImplementedError},
    prelude::*,
    types::{PyList, PyNotImplemented, PySequence, PySlice, PySliceIndices},
};
use pyochain_macros::py_abc;
use std::{
    cmp::Ordering,
    sync::{Mutex, MutexGuard, TryLockError, atomic::Ordering as AtomicOrdering},
};
use tap::prelude::*;

pub const DEFAULT_LOAD_FACTOR: usize = 1000;
pub type BoolOrNotImpl<'py> = PyResult<Either<bool, Bound<'py, PyNotImplemented>>>;
pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;

pub(super) fn try_lock_recover<'a, T>(mutex: &'a Mutex<T>, msg: &str) -> MutexGuard<'a, T> {
    match mutex.try_lock() {
        Ok(guard) => guard,
        //Recover if the guard was poisoned by an earlier panic instead of cascading.
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => panic!("{msg}"),
    }
}

#[py_abc(SortedList, SortedKeyList)]
pub(super) trait InnerSortedGetters:
    Sized + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    #[skip]
    fn get_data(&self) -> MutexGuard<'_, ListsData>;
    fn set_load(&self, load: usize);
    #[getter]
    fn get_load(&self) -> usize;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl InnerSortedGetters for $t {
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
pub(super) trait InnerSorted: InnerSortedGetters {
    #[skip]
    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    fn clear(&self) -> ();
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool>;
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
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>>;
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    #[skip]
    fn update_from_vec(&self, py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<()>;

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.collapse_lists(py);
        self.clear();
        self.set_load(load);
        self.update_from_vec(py, values)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
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
                let mut values = data.getitem_from_slice(py, PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice =
                        data.getitem_from_slice(py, PySlice::new(py, stop, length, 1))?;
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
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match slf.get().islice_specs(py, start, stop)? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }

    #[skip]
    fn islice_specs(
        &self,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<iter::IsliceBounds>> {
        let mut data = self.get_data();
        let length = data.len as isize;

        if length == 0 {
            return Ok(None);
        }
        //NOTE: Need to investiguate why we need to use PySlice at all. Same pattern in SliceView original code.
        let indices =
            PySlice::new(py, start.unwrap_or(0), stop.unwrap_or(length), 1).indices(length)?;

        if indices.start >= indices.stop {
            Ok(None)
        } else {
            let (min_pos, min_idx) = data.pos(indices.start)?;

            let (max_pos, max_idx) = if indices.stop == length {
                (
                    data.lists.len() - 1,
                    data.lists.last().unwrap().len() as isize,
                )
            } else {
                data.pos(indices.stop)?
            };

            Ok(Some(iter::IsliceBounds::new(
                min_pos,
                min_idx as usize,
                max_pos,
                max_idx as usize,
            )))
        }
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
            Either::Left(index) => {
                let mut data = self.get_data();
                let (pos, idx) = data.pos(index)?;
                self.delete(py, &mut data, pos, idx as usize)
            }
        }
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        index: Either<isize, Bound<'py, PySlice>>,
    ) -> PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>> {
        let mut data = self.get_data();
        match index {
            Either::Right(slice) => data
                .getitem_from_slice(py, slice)?
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
