use crate::collections::sorted::iter::try_iterator_into_list;
use crate::collections::sorted::traits::{
    DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters,
};
use crate::collections::sorted::{bisect, errors};
use crate::pyo3_ext::{prelude::*, pylibs};
use crate::seq::{IntoPyochain, PyoVec};
use pyo3::{prelude::*, types::PyList};
use std::sync::{Mutex, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(generic, frozen)]
pub struct InnerLists {
    #[pyo3(get)]
    pub(super) lists: Py<PyoVec>,
    #[pyo3(get)]
    pub(super) maxes: Py<PyoVec>,
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
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
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
        let maxes = self.maxes.bind(py);

        if maxes.is_empty()? {
            return Ok(None);
        }

        let lists = self.lists.bind(py);

        // Calculate the minimum (pos, idx) pair. By default this location
        // will be inclusive in our calculation.
        let (min_pos, min_idx) = match minimum {
            None => (0, 0),
            Some(minimum) => {
                if inclusive.0 {
                    let min_pos = bisect::bisect_left(maxes, &minimum)?;

                    if min_pos == maxes.len()? {
                        return Ok(None);
                    }

                    let min_idx = bisect::bisect_left(
                        &lists
                            .get_item(min_pos)
                            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?,
                        &minimum,
                    )?;
                    (min_pos, min_idx)
                } else {
                    let min_pos = bisect::bisect_right(maxes, &minimum)?;

                    if min_pos == maxes.len()? {
                        return Ok(None);
                    }

                    let min_idx = bisect::bisect_right(
                        &lists
                            .get_item(min_pos)
                            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?,
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
                    let mut pos = bisect::bisect_right(maxes, &m)?;

                    let idx = if pos == maxes.len()? {
                        pos -= 1;
                        lists.get_item(pos)?.len()?
                    } else {
                        bisect::bisect_right(
                            &lists
                                .get_item(pos)
                                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?,
                            &m,
                        )?
                    };
                    Ok::<_, PyErr>((pos, idx))
                } else {
                    let mut pos = bisect::bisect_left(maxes, &m)?;

                    let idx = if pos == maxes.len()? {
                        pos -= 1;
                        lists.get_item(pos)?.len()?
                    } else {
                        bisect::bisect_left(
                            &lists
                                .get_item(pos)
                                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?,
                            &m,
                        )?
                    };
                    Ok((pos, idx))
                }
            })
            .unwrap_or_else(|| {
                let pos = maxes.len()? - 1;
                let idx = lists.get_item(pos)?.len()?;
                Ok((pos, idx))
            })?;

        Ok(Some((min_pos, min_idx, max_pos, max_idx)))
    }
}
impl InnerSorted for InnerLists {
    fn clear(&self, py: Python<'_>) -> () {
        self.set_len(0);
        self.lists.get().clear(py);
        self.maxes.get().clear(py);
        self.get_idx().clear();
        self.set_offset(0);
    }

    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let maxes = self.maxes.bind(value.py());

        if maxes.is_empty()? {
            return Ok(false);
        }

        let pos = bisect::bisect_left(maxes, &value)?;

        if maxes.len()?.eq(&pos) {
            return Ok(false);
        }

        let lists = self.lists.bind(value.py());
        let v = &lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;

        lists.get_item(pos)?.get_item(idx)?.eq(value)
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let load = self.get_load();
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);

        if lists.get_item(pos)?.len()?.gt(&(load << 1)) {
            let maxes = self.maxes.get().inner.bind(py);

            let lists_pos = lists
                .get_item(pos)?
                .pipe(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let half = lists_pos.get_slice(load, usize::MAX);
            lists_pos.del_slice(load, usize::MAX)?;
            maxes.set_item(pos, lists_pos.last()?)?;
            let last = half.last()?;
            lists.insert(pos + 1, &half.into_pyochain()?)?;
            maxes.insert(pos + 1, last)?;

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
        let lists = self.lists.bind(py).get().inner.bind(py);
        let maxes = self.maxes.bind(py);

        let lists_pos = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);

        lists_pos.del_item(idx)?;
        self.set_len(self.get_len() - 1);

        let len_lists_pos = lists_pos.len();

        if len_lists_pos > (self.get_load() >> 1) {
            maxes.set_item(pos, lists_pos.last()?)?;

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
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(lists.get_item(pos)?)?;
            maxes.set_item(prev, lists.get_item(prev)?.get_item(-1)?)?;

            lists.del_item(pos)?;
            maxes.del_item(pos)?;
            self.get_idx().clear();

            self.expand(py, prev)
        } else if len_lists_pos != 0 {
            maxes.set_item(pos, lists_pos.last()?)?;
            Ok(())
        } else {
            lists.del_item(pos)?;
            maxes.del_item(&pos)?;
            self.get_idx().clear();
            Ok(())
        }
    }
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let lists = self.lists.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        if !maxes.is_empty() {
            let mut pos = bisect::bisect_right(self.maxes.bind(py), &value)?;

            if pos == maxes.len() {
                pos -= 1;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&value)?;
                maxes.set_item(pos, &value)?;
            } else {
                let vector = &lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

                let res = bisect::bisect_right(&vector, &value)?;
                vector.get().insert(res, &value)?;
            }

            self.expand(py, pos)?;
        } else {
            lists.append(PyList::new(py, [&value])?.into_pyochain()?)?;
            maxes.append(&value)?;
        }

        self.set_len(self.get_len() + 1);
        Ok(())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(());
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(());
        }

        let lists = self.lists.get().inner.bind(value.py());
        let v = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;

        if lists.get_item(pos)?.get_item(idx)?.eq(value)? {
            self.delete(py, pos, idx)
        } else {
            Ok(())
        }
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            errors::not_in_list_err(value)
        } else {
            let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

            if pos == maxes.len() {
                errors::not_in_list_err(value)
            } else {
                let lists = self.lists.get().inner.bind(py);
                let v = &lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

                let idx = bisect::bisect_left(&v, &value)?;

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
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_right(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

        let idx = bisect::bisect_right(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos_left = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos_left == maxes.len() {
            return Ok(0);
        }

        let lists = self.lists.get().inner.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_left = bisect::bisect_left(&v_left, &value)?;
        let pos_right = bisect::bisect_right(self.maxes.bind(py), &value)?;

        if pos_right == maxes.len() {
            return Ok(self.get_len() - self.loc(py, pos_left, idx_left as isize)? as usize);
        }
        let v_right = lists
            .get_item(pos_right)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_right = bisect::bisect_right(&v_right, &value)?;

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
            return errors::is_not_in_list_err(value);
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
            return errors::is_not_in_list_err(value);
        }

        let maxes = self.maxes.get().inner.bind(py);
        let pos_left = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos_left == maxes.len() {
            return errors::is_not_in_list_err(value);
        }

        let lists = self.lists.get().inner.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_left = bisect::bisect_left(&v_left, &value)?;

        if lists.get_item(pos_left)?.get_item(idx_left)?.ne(&value)? {
            return errors::is_not_in_list_err(value);
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

        errors::is_not_in_list_err(value)
    }

    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let maxes = self.get_maxes(py).get().inner.clone_ref(py).into_bound(py);
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?;

        if !maxes.is_empty() {
            if values.len() * 4 >= self.get_len() {
                lists.append(values.into_pyochain()?)?;
                values = self
                    .collapse_lists(py)?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py);
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

        (0..values_len)
            .step_by(load)
            .map(|pos| values.get_slice(pos, pos + load).into_pyochain())
            .try_fold(lists, try_iterator_into_list)?
            .iter()
            .map(|x| {
                unsafe { x.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .last()
            })
            .try_fold(maxes, try_iterator_into_list)?;
        self.set_len(values_len);
        self.get_idx().clear();
        Ok(())
    }
}
