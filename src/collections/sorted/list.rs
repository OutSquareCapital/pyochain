use crate::{
    abc,
    collections::sorted::{
        bisect,
        bounds::{Bounds, Pos},
        cmp::py_cmp,
        data::{ListsData, islice_list, reset_list},
        errors, iter,
        traits::{
            BaseSortedList, BaseSortedListSet, DEFAULT_LOAD_FACTOR, Reduced, SortedCollection,
            SortedListGetters,
        },
    },
    iterators,
    traits::PyoABC,
};
use pyo3::{PyTypeInfo, prelude::*, types::PyTuple};
use std::sync::{Mutex, MutexGuard, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
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
    #[inline]
    pub(super) fn into_bound<'py>(self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        abc::PyoMutableSequence::build_init()
            .add_subclass(self)
            .pipe(|x| Bound::new(py, x))
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

        abc::PyoMutableSequence::build_init()
            .add_subclass(data)
            .pipe(Ok)
    }
}
impl SortedCollection for SortedList {
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let data = self.get_data();
        let mut bound = Pos::default();

        if data.maxes.is_empty() {
            return Ok(false);
        }

        bound.pos = bisect::left(&data.maxes, &value)?;

        if data.maxes.len().eq(&bound.pos) {
            return Ok(false);
        }
        bound.idx = bisect::left(&data.lists[bound.pos], &value)?;

        data.get_value(&bound).bind(py).eq(value)
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        self.get_data()
            .as_pyovec(py)
            .and_then(|x| PyTuple::new(py, [x]))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn clear(&self, _py: Python<'_>) -> () {
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
            return errors::is_not_in_list_err(&value);
        }

        let mut start = start.unwrap_or(0);
        if start < 0 {
            start += len_;
        }
        start = start.max(0);

        let mut stop = stop.unwrap_or(len_);
        if stop < 0 {
            stop += len_;
        }
        stop = stop.min(len_);

        if stop <= start {
            return errors::is_not_in_list_err(&value);
        }
        let mut bound = Pos::default();
        bound.pos = data.maxes.pipe_ref(|maxes| {
            let pos_left = bisect::left(&maxes, &value)?;

            if pos_left == maxes.len() {
                errors::is_not_in_list_err(&value)
            } else {
                Ok(pos_left)
            }
        })?;
        bound.idx = bisect::left(&data.lists[bound.pos], &value)?;
        if data.get_value(&bound).bind(py).ne(&value)? {
            return errors::is_not_in_list_err(&value);
        }

        stop -= 1;
        let left = bound.loc(&mut data)?;

        if start <= left {
            if left <= stop {
                return Ok(left);
            }
        } else {
            drop(data);
            let right = self.bisect_right(&value)? - 1;

            if start <= right {
                return Ok(start);
            }
        }

        errors::is_not_in_list_err(&value)
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
        let mut data = self.get_data();
        if !data.maxes.is_empty() {
            let mut pos = bisect::right(&data.maxes, &value.bind(py))?;

            if pos == data.maxes.len() {
                pos -= 1;
                data.lists[pos].push(value.clone_ref(py));
                data.maxes[pos] = value;
            } else {
                let res = bisect::right(&data.lists[pos], &value.bind(py))?;
                data.lists[pos].insert(res, value.clone_ref(py));
            }
            self.expand(py, &mut data, pos)?;
        } else {
            data.lists.push(vec![value.clone_ref(py)]);
            data.maxes.push(value);
        }

        data.len = data.len + 1;
        Ok(())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();

        if data.maxes.is_empty() {
            return Ok(());
        }

        bound.pos = bisect::left(&data.maxes, &value)?;

        if bound.pos == data.maxes.len() {
            Ok(())
        } else {
            bound.idx = bisect::left(&data.lists[bound.pos], &value)?;

            if data.get_value(&bound).bind(py).eq(&value)? {
                self.delete(py, &mut data, &mut bound)
            } else {
                Ok(())
            }
        }
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();

        if data.maxes.is_empty() {
            errors::not_in_list_err(value)
        } else {
            bound.pos = bisect::left(&data.maxes, &value)?;

            if bound.pos == data.maxes.len() {
                errors::not_in_list_err(value)
            } else {
                bound.idx = bisect::left(&data.lists[bound.pos], &value)?;

                if data.get_value(&bound).bind(py).eq(&value)? {
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
        iter::SortedIter::new(py, inner)
    }
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()> {
        let load = self.get_load();

        if data.lists[pos].len().gt(&(load << 1)) {
            let half = data.lists[pos].split_off(load);
            data.maxes[pos] = data.lists[pos].last().unwrap().clone_ref(py);
            data.maxes
                .insert(pos + 1, half.last().unwrap().clone_ref(py));
            data.lists.insert(pos + 1, half);

            data.idx.clear();
            Ok(())
        } else if !data.idx.is_empty() {
            let mut child = data.offset + pos;
            while child != 0 {
                data.idx[child] = data.idx[child] + 1;
                child = (child - 1) >> 1;
            }
            data.idx[0] = data.idx[0] + 1;

            Ok(())
        } else {
            Ok(())
        }
    }

    fn delete(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        bounds: &mut Pos,
    ) -> PyResult<()> {
        data.lists[bounds.pos].remove(bounds.idx);
        data.len = data.len - 1;

        let len_lists_pos = data.lists[bounds.pos].len();

        if len_lists_pos > (self.get_load() >> 1) {
            data.maxes[bounds.pos] = data.lists[bounds.pos].last().unwrap().clone_ref(py);

            if !data.idx.is_empty() {
                let mut child = data.offset + bounds.pos;
                while child > 0 {
                    data.idx[child] = data.idx[child] - 1;
                    child = (child - 1) >> 1
                }
                data.idx[0] = data.idx[0] - 1;
            }
            Ok(())
        } else if data.lists.len() > 1 {
            if bounds.pos == 0 {
                bounds.pos += 1;
            }

            let prev = (bounds.pos - 1) as usize;
            let mut removed = data.lists[bounds.pos]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>();
            data.lists[prev].append(removed.as_mut());
            data.maxes[prev] = data.lists[prev].last().unwrap().clone_ref(py);

            data.lists.remove(bounds.pos);
            data.maxes.remove(bounds.pos);
            data.idx.clear();
            self.expand(py, data, prev)
        } else if len_lists_pos != 0 {
            data.maxes[bounds.pos] = data.lists[bounds.pos].last().unwrap().clone_ref(py);
            Ok(())
        } else {
            data.lists.remove(bounds.pos);
            data.maxes.remove(bounds.pos);
            data.idx.clear();
            Ok(())
        }
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(0);
        }
        let mut left = Pos::default();
        let mut right = Pos::default();

        left.pos = bisect::left(&data.maxes, &value)?;

        if left.pos == data.maxes.len() {
            return Ok(0);
        }
        left.idx = bisect::left(&data.lists[left.pos], &value)?;
        right.pos = bisect::right(&data.maxes, &value)?;

        if right.pos == data.maxes.len() {
            let left_loc = left.loc(&mut data)?;
            return Ok(data.len - left_loc as usize);
        }
        right.idx = bisect::right(&data.lists[right.pos], &value)?;

        if left.pos == right.pos {
            return Ok(right.idx - left.idx);
        }
        let right_loc = right.loc(&mut data)?;
        let left_loc = left.loc(&mut data)?;
        Ok((right_loc - left_loc) as usize)
    }

    fn py_update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()?
            .map(|x| x?.unbind().pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        self.update(py, values)
    }

    fn update(&self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
        values.sort_by(|a, b| py_cmp(py, a, b));
        let mut data = self.get_data();

        if !data.maxes.is_empty() {
            if values.len() * 4 >= data.len {
                data.lists.push(values);
                values = data.collapse(py);
                values.sort_by(|a, b| py_cmp(py, a, b));
                data.clear();
            } else {
                drop(data);
                for val in values {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
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
        Ok(())
    }
}
