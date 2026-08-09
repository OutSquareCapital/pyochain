use crate::{
    abc,
    collections::sorted::{
        bisect,
        cmp::py_cmp,
        data::{ListsData, get_irange_specs},
        errors, iter,
        traits::{DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters},
    },
    iterators,
    pyo3_ext::pylibs,
    pyovec::PyoVec,
    traits::{IntoPyochain, PyoABC},
};
use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyList, PyType},
};
use std::sync::MutexGuard;
use std::sync::{Mutex, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain._collections", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedList {
    pub(super) data: Mutex<ListsData>,
    pub(super) load: AtomicUsize,
}
impl SortedList {
    pub(super) fn new() -> Self {
        Self {
            data: Mutex::new(ListsData::new()),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
        }
    }
    pub(super) fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Bound<'_, Self>> {
        let new_inst = Self::new();
        new_inst.update_from_vec(py, values)?;
        abc::PyoMutableSequence::build_init()
            .add_subclass(new_inst)
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
            data.update(&values)?;
        };

        abc::PyoMutableSequence::build_init()
            .add_subclass(data)
            .pipe(Ok)
    }

    fn __reduce__<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyType>, (Bound<'py, PyoVec>,))> {
        let values = self
            .collapse_lists(py)
            .iter()
            .pipe(|x| PyList::new(py, x))?
            .into_pyochain()?;
        Ok((Self::type_object(py), (values,)))
    }
}
impl InnerSorted for SortedList {
    fn __add__<'py>(
        &self,
        py: Python<'py>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let mut values = self.collapse_lists(py);
        let mut other_vec = other
            .try_iter()?
            .map(|x| x?.unbind().clone_ref(py).pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        values.append(other_vec.as_mut());
        Self::from_vec(py, values)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        let values = self.collapse_lists(py);

        let new_values = (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>();
        Self::from_vec(py, new_values)
    }

    // @recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let cls_name = Self::type_object(py).name()?;
        self.collapse_lists(py)
            .iter()
            .pipe(|x| PyList::new(py, x))?
            .repr()
            .map(|repr| format!("{}({})", cls_name, repr))
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.collapse_lists(py))
    }

    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        iter::SortedIter::build(py, inner)
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
            .pipe(|d| get_irange_specs(&d.lists, &d.maxes, minimum, maximum, inclusive));

        match specs? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }
    fn clear(&self) -> () {
        self.get_data().clear()
    }

    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(false);
        }

        let pos = bisect::left(&data.maxes, &value)?;

        if data.maxes.len().eq(&pos) {
            return Ok(false);
        }
        let idx = bisect::left(&data.lists[pos], &value)?;

        data.lists[pos][idx].bind(py).eq(value)
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
        mut pos: usize,
        idx: usize,
    ) -> PyResult<()> {
        data.lists[pos].remove(idx);
        data.len = data.len - 1;

        let len_lists_pos = data.lists[pos].len();

        if len_lists_pos > (self.get_load() >> 1) {
            data.maxes[pos] = data.lists[pos].last().unwrap().clone_ref(py);

            if !data.idx.is_empty() {
                let mut child = data.offset + pos;
                while child > 0 {
                    data.idx[child] = data.idx[child] - 1;
                    child = (child - 1) >> 1
                }
                data.idx[0] = data.idx[0] - 1;
            }
            Ok(())
        } else if data.lists.len() > 1 {
            if pos == 0 {
                pos += 1;
            }

            let prev = (pos - 1) as usize;
            let mut removed = data.lists[pos]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>();
            data.lists[prev].append(removed.as_mut());
            data.maxes[prev] = data.lists[prev].last().unwrap().clone_ref(py);

            data.lists.remove(pos);
            data.maxes.remove(pos);
            data.idx.clear();
            self.expand(py, data, prev)
        } else if len_lists_pos != 0 {
            data.maxes[pos] = data.lists[pos].last().unwrap().clone_ref(py);
            Ok(())
        } else {
            data.lists.remove(pos);
            data.maxes.remove(pos);
            data.idx.clear();
            Ok(())
        }
    }
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

        if data.maxes.is_empty() {
            return Ok(());
        }

        let pos = bisect::left(&data.maxes, &value)?;

        if pos == data.maxes.len() {
            Ok(())
        } else {
            let idx = bisect::left(&data.lists[pos], &value)?;

            if data.lists[pos][idx].bind(py).eq(value)? {
                self.delete(py, &mut data, pos, idx)
            } else {
                Ok(())
            }
        }
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            errors::not_in_list_err(value)
        } else {
            let pos = bisect::left(&data.maxes, &value)?;

            if pos == data.maxes.len() {
                errors::not_in_list_err(value)
            } else {
                let idx = bisect::left(&data.lists[pos], &value)?;

                if data.lists[pos][idx].bind(py).eq(&value)? {
                    self.delete(py, &mut data, pos, idx)
                } else {
                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::left(&data.maxes, &value)?;

        if pos == data.maxes.len() {
            return Ok(data.len as isize);
        }
        let idx = bisect::left(&data.lists[pos], &value)?;
        data.loc(pos, idx as isize)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::right(&data.maxes, &value)?;

        if pos == data.maxes.len() {
            return Ok(data.len as isize);
        }
        let idx = bisect::right(&data.lists[pos], &value)?;
        data.loc(pos, idx as isize)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(0);
        }

        let pos_left = bisect::left(&data.maxes, &value)?;

        if pos_left == data.maxes.len() {
            return Ok(0);
        }
        let idx_left = bisect::left(&data.lists[pos_left], &value)?;
        let pos_right = bisect::right(&data.maxes, &value)?;

        if pos_right == data.maxes.len() {
            let left = data.loc(pos_left, idx_left as isize)?;
            return Ok(data.len - left as usize);
        }
        let idx_right = bisect::right(&data.lists[pos_right], &value)?;

        if pos_left == pos_right {
            return Ok(idx_right - idx_left);
        }
        let right = data.loc(pos_right, idx_right as isize)?;
        let left = data.loc(pos_left, idx_left as isize)?;
        Ok((right - left) as usize)
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
        let pos_left = data.maxes.pipe_ref(|maxes| {
            let pos_left = bisect::left(&maxes, &value)?;

            if pos_left == maxes.len() {
                errors::is_not_in_list_err(&value)
            } else {
                Ok(pos_left)
            }
        })?;
        let idx_left = bisect::left(&data.lists[pos_left], &value)?;

        if data.lists[pos_left][idx_left].bind(py).ne(&value)? {
            return errors::is_not_in_list_err(&value);
        }

        stop -= 1;
        let left = data.loc(pos_left, idx_left as isize)?;

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

    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>();
        self.update_from_vec(py, values)
    }

    fn update_from_vec(&self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
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
