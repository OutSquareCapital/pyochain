use crate::abc;
use crate::collections::sorted::iter::{self, IsliceBounds};
use crate::collections::sorted::traits::ListsData;
use crate::pyo3_ext::pylibs;
use crate::{
    collections::sorted::{
        bisect,
        cmp::py_cmp,
        errors,
        traits::{DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters},
    },
    iterators,
};
use pyo3::prelude::*;
use std::sync::MutexGuard;
use std::sync::{Mutex, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain._collections", frozen, generic)]
pub struct InnerLists {
    pub(super) data: Mutex<ListsData>,
    pub(super) len: AtomicUsize,
    pub(super) load: AtomicUsize,
    pub(super) offset: AtomicUsize,
}
impl InnerLists {
    fn irange_specs<'py>(
        &self,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<IsliceBounds>> {
        let data = self.get_data();
        let maxes = &data.maxes;

        if maxes.is_empty() {
            return Ok(None);
        }

        // Calculate the minimum (pos, idx) pair. By default this location
        // will be inclusive in our calculation.
        let (min_pos, min_idx) = match minimum {
            None => (0, 0),
            Some(minimum) => {
                if inclusive.0 {
                    let min_pos = bisect::left(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::left(&data.lists[min_pos], &minimum)?;
                    (min_pos, min_idx)
                } else {
                    let min_pos = bisect::right(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::right(&data.lists[min_pos], &minimum)?;
                    (min_pos, min_idx)
                }
            }
        };

        // Calculate the maximum (pos, idx) pair. By default this location
        // will be exclusive in our calculation.
        let (max_pos, max_idx) = maximum
            .map(|m| {
                if inclusive.1 {
                    let mut pos = bisect::right(&maxes, &m)?;

                    let idx = if pos == maxes.len() {
                        pos -= 1;
                        data.lists[pos].len()
                    } else {
                        bisect::right(&data.lists[pos], &m)?
                    };
                    Ok::<_, PyErr>((pos, idx))
                } else {
                    let mut pos = bisect::left(&maxes, &m)?;

                    let idx = if pos == maxes.len() {
                        pos -= 1;
                        data.lists[pos].len()
                    } else {
                        bisect::left(&data.lists[pos], &m)?
                    };
                    Ok((pos, idx))
                }
            })
            .unwrap_or_else(|| {
                let pos = maxes.len() - 1;
                let idx = data.lists[pos].len();
                Ok((pos, idx))
            })?;

        IsliceBounds::from_irange_spec(min_pos, min_idx, max_pos, max_idx).pipe(Ok)
    }
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            data: Mutex::new(ListsData::new()),
            len: AtomicUsize::new(0),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
            offset: AtomicUsize::new(0),
        })
    }
}
impl InnerSorted for InnerLists {
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
        match slf.get().irange_specs(minimum, maximum, inclusive)? {
            None => iterators::Iter::empty(slf.py())?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }
    fn clear(&self) -> () {
        self.set_len(0);
        let mut data = self.get_data();
        data.lists.clear();
        data.maxes.clear();
        data.idx.clear();
        self.set_offset(0);
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
            data.idx.pipe_ref_mut(|index| {
                let mut child = self.get_offset() + pos;
                while child != 0 {
                    index[child] = index[child] + 1;
                    child = (child - 1) >> 1;
                }
                index[0] = index[0] + 1;
            });
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
        self.set_len(self.get_len() - 1);

        let len_lists_pos = data.lists[pos].len();

        if len_lists_pos > (self.get_load() >> 1) {
            data.maxes[pos] = data.lists[pos].last().unwrap().clone_ref(py);

            data.idx.pipe_ref_mut(|index| {
                if !index.is_empty() {
                    let mut child = self.get_offset() + pos;
                    while child > 0 {
                        index[child] = index[child] - &1;
                        child = (child - 1) >> 1
                    }
                    index[0] = index[0] - &1;
                }
            });
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

        self.set_len(self.get_len() + 1);
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
            return Ok(self.get_len() as isize);
        }
        let idx = bisect::left(&data.lists[pos], &value)?;
        self.loc(&mut data, pos, idx as isize)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let mut data = self.get_data();

        if data.maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::right(&data.maxes, &value)?;

        if pos == data.maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let idx = bisect::right(&data.lists[pos], &value)?;
        self.loc(&mut data, pos, idx as isize)
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
            return Ok(self.get_len() - self.loc(&mut data, pos_left, idx_left as isize)? as usize);
        }
        let idx_right = bisect::right(&data.lists[pos_right], &value)?;

        if pos_left == pos_right {
            return Ok(idx_right - idx_left);
        }
        let right = self.loc(&mut data, pos_right, idx_right as isize)?;
        let left = self.loc(&mut data, pos_left, idx_left as isize)?;
        Ok((right - left) as usize)
    }

    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let len_ = self.get_len() as isize;

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
        let mut data = self.get_data();
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
        let left = self.loc(&mut data, pos_left, idx_left as isize)?;

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
        let mut values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>();

        if !self.get_data().maxes.is_empty() {
            if values.len() * 4 >= self.get_len() {
                let mut data = self.get_data();
                data.lists.push(values);
                values = data.collapse(py);
                values.sort_by(|a, b| py_cmp(py, a, b));
                drop(data);
                self.clear();
            } else {
                for val in values {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let mut data = self.get_data();
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
        self.set_len(values_len);
        data.idx.clear();
        Ok(())
    }

    fn update_from_vec(&self, py: Python<'_>, mut iterable: Vec<Py<PyAny>>) -> PyResult<()> {
        let mut data = self.get_data();
        iterable.sort_by(|a, b| py_cmp(py, a, b));

        if !data.maxes.is_empty() {
            if iterable.len() * 4 >= self.get_len() {
                data.lists.push(iterable);
                iterable = data.collapse(py);
                iterable.sort_by(|a, b| py_cmp(py, a, b));
                self.clear();
            } else {
                for val in iterable {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let values_len = iterable.len();
        let new_lists = (0..values_len).step_by(load).map(|pos| {
            iterable[pos..(pos + load).min(values_len)]
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
        self.set_len(values_len);
        data.idx.clear();
        Ok(())
    }
}
