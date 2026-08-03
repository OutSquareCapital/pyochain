use crate::collections::sorted::iter::iterator_into_list;
use crate::collections::sorted::traits::{
    DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters,
};
use crate::collections::sorted::{bisect, errors};
use crate::pyo3_ext::{prelude::*, pylibs};
use crate::traits::PyWrapper;
use pyo3::{prelude::*, types::PyList};
use std::sync::{Mutex, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain._collections", frozen, generic)]
pub struct InnerLists {
    #[pyo3(get)]
    pub(super) lists: Py<PyList>,
    pub(super) maxes: Mutex<Vec<Py<PyAny>>>,
    pub(super) idx: Mutex<Vec<usize>>,
    pub(super) len: AtomicUsize,
    pub(super) load: AtomicUsize,
    pub(super) offset: AtomicUsize,
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            lists: PyList::empty(py).into(),
            maxes: Mutex::new(Vec::new()),
            idx: Mutex::new(Vec::new()),
            len: AtomicUsize::new(0),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
            offset: AtomicUsize::new(0),
        })
    }
    #[pyo3(signature = (minimum=None, maximum=None, inclusive=(true, true)))]
    fn irange(
        &self,
        py: Python<'_>,
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<(usize, usize, usize, usize)>> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(None);
        }

        let lists = self.lists.bind(py);

        // Calculate the minimum (pos, idx) pair. By default this location
        // will be inclusive in our calculation.
        let (min_pos, min_idx) = match minimum {
            None => (0, 0),
            Some(minimum) => {
                if inclusive.0 {
                    let min_pos = bisect::left_vec(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::left(
                        &lists
                            .get_item(min_pos)
                            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
                        &minimum,
                    )?;
                    (min_pos, min_idx)
                } else {
                    let min_pos = bisect::right_vec(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::right(
                        &lists
                            .get_item(min_pos)
                            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
                        &minimum,
                    )?;
                    (min_pos, min_idx)
                }
            }
        };

        // Calculate the maximum (pos, idx) pair. By default this location
        // will be exclusive in our calculation.
        let (max_pos, max_idx) = maximum
            .map(|m| {
                if inclusive.1 {
                    let mut pos = bisect::right_vec(&maxes, &m)?;

                    let idx = if pos == maxes.len() {
                        pos -= 1;
                        lists.get_item(pos)?.len()?
                    } else {
                        bisect::right(
                            &lists
                                .get_item(pos)
                                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
                            &m,
                        )?
                    };
                    Ok::<_, PyErr>((pos, idx))
                } else {
                    let mut pos = bisect::left_vec(&maxes, &m)?;

                    let idx = if pos == maxes.len() {
                        pos -= 1;
                        lists.get_item(pos)?.len()?
                    } else {
                        bisect::left(
                            &lists
                                .get_item(pos)
                                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
                            &m,
                        )?
                    };
                    Ok((pos, idx))
                }
            })
            .unwrap_or_else(|| {
                let pos = maxes.len() - 1;
                let idx = lists.get_item(pos)?.len()?;
                Ok((pos, idx))
            })?;

        Ok(Some((min_pos, min_idx, max_pos, max_idx)))
    }
}
impl InnerSorted for InnerLists {
    fn clear(&self, py: Python<'_>) -> () {
        self.set_len(0);
        self.lists.bind(py).clear();
        self.get_maxes().clear();
        self.get_idx().clear();
        self.set_offset(0);
    }

    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(false);
        }

        let pos = bisect::left_vec(&maxes, &value)?;

        if maxes.len().eq(&pos) {
            return Ok(false);
        }

        let lists = self.lists.bind(py);
        let v = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
        let idx = bisect::left(&v, &value)?;

        lists.get_item(pos)?.get_item(idx)?.eq(value)
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let load = self.get_load();
        let lists = self.lists.clone_ref(py).into_bound(py);

        if lists.get_item(pos)?.len()?.gt(&(load << 1)) {
            let mut maxes = self.get_maxes();

            let lists_pos = lists
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
            let half = lists_pos.get_slice(load, usize::MAX);
            lists_pos.del_slice(load, usize::MAX)?;
            maxes[pos] = lists_pos.last()?.unbind();
            let last = half.last()?;
            lists.insert(pos + 1, &half)?;
            maxes.insert(pos + 1, last.unbind());

            self.get_idx().clear();
            Ok(())
        } else if !self.get_idx().is_empty() {
            self.get_idx().pipe_ref_mut(|index| {
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

    fn delete(&self, py: Python<'_>, mut pos: usize, idx: usize) -> PyResult<()> {
        let lists = self.lists.bind(py);
        let mut maxes = self.get_maxes();

        let lists_pos = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;

        lists_pos.del_item(idx)?;
        self.set_len(self.get_len() - 1);

        let len_lists_pos = lists_pos.len();

        if len_lists_pos > (self.get_load() >> 1) {
            maxes[pos] = lists_pos.last()?.unbind();

            self.get_idx().pipe_ref_mut(|index| {
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
        } else if lists.len() > 1 {
            if pos == 0 {
                pos += 1;
            }

            let prev = (pos - 1) as usize;
            lists
                .get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?
                .extend(lists.get_item(pos)?)?;
            maxes[prev] = lists.get_item(prev)?.get_item(-1)?.unbind();

            lists.del_item(pos)?;
            maxes.remove(pos);
            self.get_idx().clear();
            drop(maxes);
            self.expand(py, prev)
        } else if len_lists_pos != 0 {
            maxes[pos] = lists_pos.last()?.unbind();
            Ok(())
        } else {
            lists.del_item(pos)?;
            maxes.remove(pos);
            self.get_idx().clear();
            Ok(())
        }
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let lists = self.lists.bind(py);
        let mut maxes = self.get_maxes();
        if !maxes.is_empty() {
            let mut pos = bisect::right_vec(&maxes, &value)?;

            if pos == maxes.len() {
                pos -= 1;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?
                    .append(&value)?;
                maxes[pos] = value.unbind();
            } else {
                let vector = lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;

                let res = bisect::right(&vector, &value)?;
                vector.insert(res, &value)?;
            }
            drop(maxes);
            self.expand(py, pos)?;
        } else {
            lists.append(PyList::new(py, [&value])?)?;
            maxes.push(value.unbind());
        }

        self.set_len(self.get_len() + 1);
        Ok(())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(());
        }

        let pos = bisect::left_vec(&maxes, &value)?;

        if pos == maxes.len() {
            Ok(())
        } else {
            drop(maxes);
            let lists = self.lists.bind(value.py());
            let v = lists
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
            let idx = bisect::left(&v, &value)?;

            if lists.get_item(pos)?.get_item(idx)?.eq(value)? {
                self.delete(py, pos, idx)
            } else {
                Ok(())
            }
        }
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            errors::not_in_list_err(value)
        } else {
            let pos = bisect::left_vec(&maxes, &value)?;

            if pos == maxes.len() {
                errors::not_in_list_err(value)
            } else {
                drop(maxes);
                let lists = self.lists.bind(py);
                let v = lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;

                let idx = bisect::left(&v, &value)?;

                if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                    self.delete(py, pos, idx)
                } else {
                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::left_vec(&maxes, &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
        let idx = bisect::left(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::right_vec(&maxes, &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;

        let idx = bisect::right(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos_left = bisect::left_vec(&maxes, &value)?;

        if pos_left == maxes.len() {
            return Ok(0);
        }

        let lists = self.lists.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
        let idx_left = bisect::left(&v_left, &value)?;
        let pos_right = bisect::right_vec(&maxes, &value)?;

        if pos_right == maxes.len() {
            return Ok(self.get_len() - self.loc(py, pos_left, idx_left as isize)? as usize);
        }
        let v_right = lists
            .get_item(pos_right)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
        let idx_right = bisect::right(&v_right, &value)?;

        if pos_left == pos_right {
            return Ok(idx_right - idx_left);
        }

        let right = self.loc(py, pos_right, idx_right as isize)?;
        let left = self.loc(py, pos_left, idx_left as isize)?;
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

        let pos_left = self.get_maxes().pipe(|maxes| {
            let pos_left = bisect::left_vec(&maxes, &value)?;

            if pos_left == maxes.len() {
                errors::is_not_in_list_err(&value)
            } else {
                Ok(pos_left)
            }
        })?;

        let lists = self.lists.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?;
        let idx_left = bisect::left(&v_left, &value)?;

        if lists.get_item(pos_left)?.get_item(idx_left)?.ne(&value)? {
            return errors::is_not_in_list_err(&value);
        }

        stop -= 1;
        let left = self.loc(py, pos_left, idx_left as isize)?;

        if start <= left {
            if left <= stop {
                return Ok(left);
            }
        } else {
            let right = self.bisect_right(&value)? - 1;

            if start <= right {
                return Ok(start);
            }
        }

        errors::is_not_in_list_err(&value)
    }

    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let lists = self.get_lists(py).clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?;

        if !self.get_maxes().is_empty() {
            if values.len() * 4 >= self.get_len() {
                lists.append(values)?;
                values = self.collapse_lists(py)?.get().into_inner_bound(py);
                values.sort()?;
                self.clear(py);
            } else {
                for val in values {
                    self.add(val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let values_len = values.len();

        let mut new_maxes = (0..values_len)
            .step_by(load)
            .map(|pos| values.get_slice(pos, pos + load))
            .try_fold(lists, iterator_into_list)?
            .iter()
            .map(|x| {
                unsafe { x.cast_into_unchecked::<PyList>() }
                    .last()
                    .map(Bound::unbind)
            })
            .collect::<PyResult<Vec<_>>>()?;
        self.get_maxes().append(new_maxes.as_mut());
        self.set_len(values_len);
        self.get_idx().clear();
        Ok(())
    }
}
