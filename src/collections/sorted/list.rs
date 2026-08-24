use crate::{
    abc,
    collections::sorted::{
        bisect,
        bounds::{Bounds, Indexes, Pos},
        cmp::py_cmp,
        data::{ListsData, islice_list, reset_list},
        errors, iter, ops,
        traits::{
            BaseSortedList, BaseSortedListSet, DEFAULT_LOAD_FACTOR, Reduced, SortedCollection,
            SortedListGetters,
        },
    },
    core::iterators,
    traits::IntoInit,
};
use pyo3::{PyTypeInfo, prelude::*, types::PyTuple};
use std::sync::{Mutex, MutexGuard, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain.collections._sorted", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList {
    pub(super) data: Mutex<ListsData>,
    pub(super) load: AtomicUsize,
}
impl SortedList {
    #[inline]
    pub(super) fn new() -> Self {
        Self {
            data: Mutex::new(ListsData::default()),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
        }
    }
    #[inline]
    pub(super) fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Self> {
        let new_inst = Self::new();
        new_inst.update(py, values).map(|_| new_inst)
    }
}
#[pymethods]
impl SortedList {
    #[new]
    #[pyo3(signature = (iterable = None))]
    fn py_new(iterable: Option<Bound<'_, PyAny>>) -> PyResult<PyClassInitializer<Self>> {
        let data = Self::new();
        if let Some(values) = iterable {
            data.py_update(&values)?;
        };

        data.init().pipe(Ok)
    }
}
impl SortedCollection for SortedList {
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let data = self.get_data();
        let mut bound = Pos::default();
        match ops::Maxes::new(&data.maxes, &mut bound, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(false),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&data.lists[bound.pos], value)?;
                data.get_value(&bound).bind(py).eq(value)
            }
        }
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        self.get_data()
            .as_pyovec(py)
            .and_then(|x| PyTuple::new(py, [x]))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn clear(&self, _py: Python<'_>) {
        self.get_data().clear()
    }

    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_left(None, value)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_right(None, value)
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();

        let mut data = self.get_data();
        let len_ = data.len as isize;

        if len_ == 0 {
            errors::not_in_list_err(&value)
        } else {
            let mut indexes = Indexes::new(start, stop, len_);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(&value)
            } else {
                let mut bound = Pos {
                    pos: bisect::left(&data.maxes, &value)?,
                    idx: 0,
                };
                if bound.pos == data.maxes.len() {
                    errors::not_in_list_err(&value)
                } else {
                    bound.idx = bisect::left(&data.lists[bound.pos], &value)?;
                    if data.get_value(&bound).bind(py).ne(&value)? {
                        errors::not_in_list_err(&value)
                    } else {
                        indexes.stop -= 1;
                        let left = bound.loc(&mut data)?;

                        if indexes.start <= left {
                            if left <= indexes.stop {
                                return Ok(left);
                            }
                        } else {
                            drop(data);
                            let right = self.bisect_right(&value)? - 1;

                            if indexes.start <= right {
                                return Ok(indexes.start);
                            }
                        }
                        errors::not_in_list_err(&value)
                    }
                }
            }
        }
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        reset_list(self, py, load)
    }
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let specs = slf
            .get()
            .get_data()
            .pipe(|d| Bounds::get_irange_specs(&d.lists, &d.maxes, minimum, maximum, inclusive));

        match specs? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }

    fn islice<'py>(
        slf: Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        islice_list(slf, start, stop, reverse)
    }
}
impl BaseSortedListSet for SortedList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let mut bound = Pos::default();
        let mut data = self.get_data();
        match ops::Maxes::new(&data.maxes, &mut bound, value.bind(py), bisect::right)? {
            ops::Maxes::Empty => {
                data.lists.push(vec![value.clone_ref(py)]);
                data.maxes.push(value);
            }
            ops::Maxes::LenEQPos => {
                bound.pos -= 1;
                data.lists[bound.pos].push(value.clone_ref(py));
                data.maxes[bound.pos] = value;
                self.expand(py, &mut data, bound.pos)?;
            }
            ops::Maxes::LenNEPos => {
                let res = bisect::right(&data.lists[bound.pos], value.bind(py))?;
                data.lists[bound.pos].insert(res, value.clone_ref(py));
                self.expand(py, &mut data, bound.pos)?;
            }
        }
        data.len += 1;
        Ok(())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();
        match ops::Maxes::new(&data.maxes, &mut bound, &value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(()),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&data.lists[bound.pos], &value)?;
                if data.get_value(&bound).bind(py).eq(&value)? {
                    self.delete(py, &mut data, &mut bound)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();
        match ops::Maxes::new(&data.maxes, &mut bound, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => errors::not_in_list_err(value),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&data.lists[bound.pos], value)?;
                if data.get_value(&bound).bind(py).eq(value)? {
                    self.delete(py, &mut data, &mut bound)
                } else {
                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().collapse(py))?.into_bound(py)
    }
}
impl BaseSortedList for SortedList {
    fn __add__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let data = slf.get().get_data();
        let out = if other.is(&slf) {
            data.repeat(py, 2)
        } else {
            data.concat(py, other)?
        };
        Self::from_vec(py, out)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().repeat(py, num))?.into_bound(py)
    }

    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let cls_name = Self::type_object(py).name()?;
        self.get_data()
            .py_repr(py)
            .map(|repr| format!("{}({})", cls_name, repr))
    }

    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        iter::SortedIter::new(inner)
            .into_bound(py)
            .map(Bound::into_super)
    }
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()> {
        let load = self.get_load();
        match ops::Expand::new(data.lists[pos].len(), load, &data.idx) {
            ops::Expand::PosLenGtLoad => {
                let half = data.lists[pos].split_off(load);
                let new_max_at_pos = data.lists[pos].last().unwrap().clone_ref(py);
                let last_max = half.last().unwrap().clone_ref(py);
                data.expand_at_pos(pos, half, last_max, new_max_at_pos);
                Ok(())
            }
            ops::Expand::IdxNotEmpty => data.expand_on_empty_idx(pos),
            ops::Expand::Other => Ok(()),
        }
    }

    fn delete(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        bounds: &mut Pos,
    ) -> PyResult<()> {
        data.lists[bounds.pos].remove(bounds.idx);
        data.len -= 1;
        match ops::Delete::new(&data.lists, self.get_load(), bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = data.lists[bounds.pos].last().unwrap().clone_ref(py);
                data.delete_on_idx(bounds, max_at_pos)
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1;
                }
                let prev = bounds.pos - 1;
                data.set_prev_from_removed(py, bounds, prev);
                data.maxes[prev] = data.lists[prev].last().unwrap().clone_ref(py);
                self.expand(py, data, prev)?
            }
            ops::Delete::LenPosNotZero => {
                data.maxes[bounds.pos] = data.lists[bounds.pos].last().unwrap().clone_ref(py)
            }
            ops::Delete::Other => data.remove_pos(bounds),
        };
        Ok(())
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let mut data = self.get_data();
        let mut left = Pos::default();
        match ops::Maxes::new(&data.maxes, &mut left, &value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let mut right = Pos::default();
                left.idx = bisect::left(&data.lists[left.pos], &value)?;
                right.pos = bisect::right(&data.maxes, &value)?;

                if right.pos == data.maxes.len() {
                    let left_loc = left.loc(&mut data)?;
                    Ok(data.len - left_loc as usize)
                } else {
                    right.idx = bisect::right(&data.lists[right.pos], &value)?;

                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_loc = right.loc(&mut data)?;
                        let left_loc = left.loc(&mut data)?;
                        Ok((right_loc - left_loc) as usize)
                    }
                }
            }
        }
    }

    fn update(&self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
        values.sort_by(|a, b| py_cmp(py, a, b));
        let mut data = self.get_data();
        let load = self.get_load();
        match ops::Update::new(&data.maxes, data.len, &values) {
            ops::Update::EmptyMaxes => finalize_update(load, py, values, &mut data),
            ops::Update::OtherGESelf => {
                data.lists.push(values);
                values = data.collapse(py);
                values.sort_by(|a, b| py_cmp(py, a, b));
                data.clear();
                finalize_update(load, py, values, &mut data)
            }
            ops::Update::OtherLTSelf => {
                drop(data);
                for val in values {
                    self.add(py, val)?;
                }
            }
        };
        Ok(())
    }
}

fn finalize_update(
    load: usize,
    py: Python<'_>,
    values: Vec<Py<PyAny>>,
    data: &mut MutexGuard<'_, ListsData>,
) {
    let values_len = values.len();
    let new_lists = (0..values_len).step_by(load).map(|pos| {
        values[pos..(pos + load).min(values_len)]
            .iter()
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>()
    });
    data.lists.extend(new_lists);
    let mut new_maxes = data
        .lists
        .iter()
        .map(|x| x.last().unwrap().clone_ref(py))
        .collect::<Vec<_>>();
    data.maxes.append(new_maxes.as_mut());
    data.len = values_len;
    data.idx.clear();
}
