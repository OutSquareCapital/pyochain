use crate::{
    abc,
    collections::{
        SortedKeyList, SortedList,
        sorted::{
            bounds::{Bounds, Pos},
            data::ListsData,
            dict::{SortedDict, SortedKeyDict},
            iter,
            keyset::SortedKeySet,
            set::SortedSet,
            views::BaseSortedView,
        },
    },
    core::PyoVec,
    traits::IntoPyochain,
};
use either::Either;
use pyo3::{
    PyClass, PyTypeInfo,
    call::PyCallArgs,
    exceptions::{PyIndexError, PyKeyError, PyNotImplementedError},
    prelude::*,
    types::{
        PyBool, PyDict, PyList, PyMapping, PyNotImplemented, PySequence, PySet, PySlice,
        PySliceIndices, PyTuple, PyType,
    },
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use pyochain_macros::{py_abc, try_cast, try_cast_into};
use std::{
    cmp::Ordering,
    sync::{Mutex, MutexGuard, TryLockError, atomic::Ordering as AtomicOrdering},
};
use tap::prelude::*;

pub const DEFAULT_LOAD_FACTOR: usize = 1000;
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
#[py_abc(
    SortedList,
    SortedKeyList,
    SortedSet,
    SortedKeySet,
    SortedDict,
    SortedKeyDict
)]
pub(super) trait SortedCollection:
    Sized + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py>;
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool>;
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
    fn clear(&self, py: Python<'_>) -> ();
}

#[py_abc(SortedList, SortedKeyList, SortedSet, SortedKeySet)]
pub(super) trait BaseSortedListSet: SortedCollection {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()>;
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
        bounds: &mut Pos,
    ) -> PyResult<()>;
    #[skip]
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()>;
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    #[pyo3(name = "update")]
    fn py_update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.update(py, values)
    }
    #[skip]
    fn update(&self, py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<()>;
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let mut data = self.get_data();
        let mut bounds = Pos::default();
        if data.len == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }
        let len_last = data.lists.last().unwrap().len() as isize;
        match index {
            -1 => {
                bounds.pos = data.lists.len() - 1;
                bounds.idx = data.lists[bounds.pos].len() - 1 as usize;
            }
            _ if 0 <= index && index < data.lists[0].len() as isize => {
                bounds.idx = index as usize;
            }
            _ if -len_last < index && index < 0 => {
                bounds.pos = data.lists.len() - 1;
                bounds.idx = (len_last + index) as usize;
            }
            _ => {
                bounds.set_from_pos(index, &mut data)?;
            }
        };
        let val = data.get_value(&bounds).clone_ref(py);
        self.delete(py, &mut data, &mut bounds)?;
        Ok(val.into_bound(py))
    }
    #[skip]
    fn delitem_from_slice(&self, py: Python<'_>, slice: Bound<'_, PySlice>) -> PyResult<()> {
        let mut data = self.get_data();
        let length = data.len as isize;
        let mut bounds = Pos::default();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step, start.cmp(&stop)) {
            (1, Ordering::Less) if start == 0 && stop == length => {
                drop(data);
                self.clear(py);
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
                self.clear(py);
                self.update(py, values)?;
                Ok(())
            }
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .rev()
                .try_for_each(|idx| {
                    bounds.set_from_pos(idx, &mut data)?;
                    self.delete(py, &mut data, &mut bounds)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        bounds.set_from_pos(idx, &mut data)?;
                        self.delete(py, &mut data, &mut bounds)
                    })
            }
        }
    }
    fn delitem_from_int(&self, py: Python<'_>, index: isize) -> PyResult<()> {
        let mut data = self.get_data();
        let mut bounds = Pos::default();
        bounds.set_from_pos(index, &mut data)?;
        self.delete(py, &mut data, &mut bounds)
    }
    /// Return an iterator that slices sorted list using two index pairs.\
    /// The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the first inclusive and the latter exclusive.\
    /// See `_pos` for details on how an index is converted to an index pair.\
    /// When `reverse` is `True`, values are yielded from the iterator in reverse order.
    #[skip]
    fn islice_iter<'py>(
        slf: Bound<'py, Self>,
        bounds: Bounds,
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
    fn __eq__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        let data = self.get_data();
        match other {
            Either::Left(seq) => {
                if data.len.ne(&seq.len()?) {
                    Either::Left(false).pipe(Ok)
                } else {
                    let py = seq.py();
                    data.iter()
                        .zip(seq.iter_py())
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

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    fn __ne__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        let data = self.get_data();
        match other {
            Either::Left(seq) => {
                if data.len.ne(&seq.len()?) {
                    Either::Left(true).pipe(Ok)
                } else {
                    let py = seq.py();
                    data.iter()
                        .zip(seq.iter_py())
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
            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    fn __lt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.iter_py()) {
                    let a = alpha.bind(py);
                    let b = beta?;
                    if a.ne(&b)? {
                        return a.lt(&b).map(Either::Left);
                    }
                }

                data.len.lt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    fn __gt__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return Either::Left(a.gt(&b)?).pipe(Ok);
                    }
                }
                data.len.gt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    fn __le__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.le(b).map(Either::Left);
                    }
                }

                data.len.le(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    fn __ge__<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<bool, 'py> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                let data = self.get_data();
                for (alpha, beta) in data.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.ge(b).map(Either::Left);
                    }
                }

                data.len.ge(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }
            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
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
        self.py_update(&other)
    }

    fn __imul__(&self, py: Python<'_>, num: usize) -> PyResult<()> {
        let values = self.get_data().repeat(py, num);
        self.clear(py);
        self.update(py, values)
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
pub(super) trait ListGetter {
    type T: BaseSortedList;
    fn get_list(&self) -> &Py<Self::T>;
    #[inline(always)]
    fn get_list_bound<'py>(&self, py: Python<'py>) -> Bound<'py, Self::T> {
        self.get_list().clone_ref(py).into_bound(py)
    }
}

#[py_abc(SortedSet, SortedKeySet)]
pub(super) trait BaseSortedSet: ListGetter + BaseSortedListSet {
    #[inline(always)]
    #[skip]
    fn from_set<'py>(&self, values: Bound<'py, PySet>) -> PyResult<Bound<'py, Self>>;
    #[getter]
    fn get_set(&self) -> &Py<PySet>;
    fn __repr__(&self, py: Python<'_>) -> PyResult<String>;

    #[skip]
    fn update<'py>(&self, py: Python<'py>, other: IntoUpdate<'py>) -> PyResult<()> {
        let set = self.get_set().bind(py);

        let values = other.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.update((values,))?;
            let list = self.get_list().get();
            list.clear(py);
            list.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        } else {
            for value in values.iter().map(Bound::unbind) {
                self.add(py, value)?;
            }
        };
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
            .and_then(|diff| self.from_set(diff))
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
            .and_then(|intersect| self.from_set(intersect))
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
            .and_then(|u| self.from_set(u))
    }
    #[skip]
    fn difference_update<'py>(&self, py: Python<'_>, iterables: IntoUpdate<'py>) -> PyResult<()> {
        let set = self.get_set().bind(py);
        let values = iterables.into_set(py)?;
        if (4 * values.len()) > set.len() {
            set.difference_update((values,))?;
            let list = self.get_list().get();
            list.clear(py);
            list.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        } else {
            for value in values {
                self.discard(value)?
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
        let list = self.get_list().get();
        list.clear(py);
        list.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, index: IntOrSlice<'py>) -> ObjOrVec<'py> {
        self.get_list().get().__getitem__(py, index)
    }
    fn __delitem__<'py>(&self, py: Python<'_>, index: IntOrSlice<'py>) -> PyResult<()> {
        match index {
            Either::Right(slice) => {
                let values = self
                    .get_list()
                    .get()
                    .get_data()
                    .getitem_from_slice(py, &slice)?
                    .iter()
                    .collect_bound::<PySet>(py)?;
                self.get_set().bind(py).difference_update((values,))?;
                self.get_list().get().delitem_from_slice(py, slice)?;
            }
            Either::Left(int) => {
                let value = self.get_list().get().get_data().getitem_from_int(py, int)?;
                self.get_set().bind(py).remove(&value)?;
                self.get_list().get().delitem_from_int(py, int)?;
            }
        };
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

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.get_list_bound(py).pipe(Self::T::__iter__)
    }

    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.get_list_bound(py).pipe(Self::T::__reversed__)
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }
    fn __sub__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.difference(other.py(), (other,))
    }
    fn __isub__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<()> {
        slf.get()
            .difference_update(slf.py(), IntoUpdate::from_any(other))
            .map(|_| ())
    }

    fn __and__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.intersection(other.py(), (other,))
    }
    fn __rand__<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.__and__(other)
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<()> {
        slf.get()
            .intersection_update(slf.py(), (other,))
            .map(|_| ())
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
        let value = self.get_list().get().pop(py, index)?;
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
            let list = slf_ref.get_list().get();
            list.clear(py);
            list.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        } else {
            for value in values {
                slf_ref.discard(value)?
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
            .map(|_| slf)
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
        let py = other.py();
        let slf_ref = slf.get();
        let set = slf_ref.get_set().bind(other.py());
        set.symmetric_difference_update(other)?;
        let list = slf_ref.get_list().get();
        list.clear(py);
        list.update(py, set.iter().map(Bound::unbind).collect::<Vec<_>>())?;
        Ok(slf)
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
            .map(|_| slf)
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

            fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
                self.get_list().get().bisect_left(value)
            }

            fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
                self.get_list().get().bisect_right(value)
            }

            fn index(
                &self,
                value: Bound<'_, PyAny>,
                start: Option<isize>,
                stop: Option<isize>,
            ) -> PyResult<isize> {
                self.get_list().get().index(value, start, stop)
            }
            #[allow(unused_variables)]
            fn islice<'py>(
                slf: Bound<'py, Self>,
                start: Option<isize>,
                stop: Option<isize>,
                reverse: bool,
            ) -> PyResult<Bound<'py, abc::PyoIterator>> {
                slf.get()
                    .get_list_bound(slf.py())
                    .pipe(|list| <$list>::islice(list, start, stop, reverse))
            }
            #[allow(unused_variables)]
            fn irange<'py>(
                slf: Bound<'py, Self>,
                minimum: Option<Bound<'py, PyAny>>,
                maximum: Option<Bound<'py, PyAny>>,
                inclusive: (bool, bool),
                reverse: bool,
            ) -> PyResult<Bound<'py, abc::PyoIterator>> {
                slf.get()
                    .get_list_bound(slf.py())
                    .pipe(|list| <$list>::irange(list, minimum, maximum, inclusive, reverse))
            }

            fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
                self.get_list().get().reset(py, load)
            }
            fn clear(&self, py: Python<'_>) -> () {
                self.get_set().bind(py).clear();
                self.get_list().get().clear(py)
            }
        }
        impl BaseSortedListSet for $set {
            fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(py);
                if !set.contains(&value)? {
                    set.add(&value)?;
                    self.get_list().get().add(py, value)?;
                }
                Ok(())
            }
            fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
                let set = self.get_set().bind(value.py());
                if set.contains(&value)? {
                    set.remove(&value)?;
                    self.get_list().get().remove(&value)?;
                }
                Ok(())
            }

            fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
                self.get_set().bind(value.py()).remove(&value)?;
                self.get_list().get().remove(value)
            }
            fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
                PySet::new(py, self.get_set().bind(py).iter()).and_then(|x| self.from_set(x))
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
    fn __or__<'py>(&self, value: Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>>;
    fn __ror__<'py>(&self, value: Bound<'py, PyMapping>) -> PyResult<Bound<'py, Self>>;
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
    fn iter<'py>(&self, py: Python<'py>) -> SortedDictIter<'_, 'py> {
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
        self.get_list().get().remove(&key)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.get_list_bound(py).pipe(Self::T::__iter__)
    }

    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, abc::PyoIterator>> {
        self.get_list_bound(py).pipe(Self::T::__reversed__)
    }

    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = key.py();
        if !self.__contains__(&key)? {
            self.get_list().get().add(py, key.clone().unbind())?;
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
            self.get_list().get().remove(&key)?;
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
            let key = self.get_list().get().pop(py, index)?;
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
        let key = self
            .get_list()
            .get()
            .get_data()
            .getitem_from_int(py, index)?;
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
            self.get_list().get().add(py, key.unbind())?;
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
        let list = self.get_list().get();
        let inner = self.get_inner().bind(py);
        if self.len(py) == 0 {
            if let Some(it) = m {
                try_cast! {match it {
                    CaseExact::PyDict(d) => inner.update(d.as_mapping())?,
                    Case::PyMapping(m) => inner.update(m)?,
                    iterable => {inner.update_from_sequence(&iterable)?;}
                }}
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
                list.clear(py);
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

pub(super) struct SortedDictIter<'a, 'py> {
    py: Python<'py>,
    mapping: Bound<'py, PyAny>,
    mapping_list: MutexGuard<'a, ListsData>,
    range: std::ops::Range<isize>,
}
impl<'a, 'py> SortedDictIter<'a, 'py> {
    fn new<D: BaseSortedDict>(owner: &'a D, py: Python<'py>) -> Self {
        let mapping = owner.get_inner().clone_ref(py).into_bound(py).into_any();
        let mapping_list = owner.get_list().get().get_data();
        let range = 0..mapping_list.len as isize;
        Self {
            py,
            mapping,
            mapping_list,
            range,
        }
    }
}
impl<'a, 'py> Iterator for SortedDictIter<'a, 'py> {
    type Item = PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>;
    fn next(&mut self) -> Option<PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
        let index = match self.range.next() {
            Some(i) => i,
            None => return None,
        };
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
