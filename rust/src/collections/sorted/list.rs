use crate::collections::sorted::traits::{
    DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters,
};
use crate::collections::sorted::{bisect, errors};
use crate::pyo3_ext::pylibs;
use pyo3::prelude::*;
use std::sync::{Mutex, atomic::AtomicUsize};

use tap::prelude::*;
#[pyclass(module = "pyochain._collections", frozen, generic)]
pub struct InnerLists {
    pub(super) lists: Mutex<Vec<Vec<Py<PyAny>>>>,
    pub(super) maxes: Mutex<Vec<Py<PyAny>>>,
    pub(super) idx: Mutex<Vec<usize>>,
    pub(super) len: AtomicUsize,
    pub(super) load: AtomicUsize,
    pub(super) offset: AtomicUsize,
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            lists: Mutex::new(Vec::new()),
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
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<(usize, usize, usize, usize)>> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(None);
        }

        let lists = self.get_lists();

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

                    let min_idx = bisect::left(&lists[min_pos], &minimum)?;
                    (min_pos, min_idx)
                } else {
                    let min_pos = bisect::right(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::right(&lists[min_pos], &minimum)?;
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
                        lists[pos].len()
                    } else {
                        bisect::right(&lists[pos], &m)?
                    };
                    Ok::<_, PyErr>((pos, idx))
                } else {
                    let mut pos = bisect::left(&maxes, &m)?;

                    let idx = if pos == maxes.len() {
                        pos -= 1;
                        lists[pos].len()
                    } else {
                        bisect::left(&lists[pos], &m)?
                    };
                    Ok((pos, idx))
                }
            })
            .unwrap_or_else(|| {
                let pos = maxes.len() - 1;
                let idx = lists[pos].len();
                Ok((pos, idx))
            })?;

        Ok(Some((min_pos, min_idx, max_pos, max_idx)))
    }
}
impl InnerSorted for InnerLists {
    fn clear(&self) -> () {
        self.set_len(0);
        self.get_lists().clear();
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

        let pos = bisect::left(&maxes, &value)?;

        if maxes.len().eq(&pos) {
            return Ok(false);
        }

        let lists = self.get_lists();
        let idx = bisect::left(&lists[pos], &value)?;

        lists[pos][idx].bind(py).eq(value)
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let load = self.get_load();
        let mut lists = self.get_lists();

        if lists[pos].len().gt(&(load << 1)) {
            let mut maxes = self.get_maxes();

            let half = &lists[pos].drain(load..usize::MAX).collect::<Vec<_>>();
            maxes[pos] = lists[pos].last().unwrap().clone_ref(py);
            let last = half.last().unwrap();
            lists.insert(
                pos + 1,
                half.iter().map(|x| x.clone_ref(py)).collect::<Vec<_>>(),
            );
            maxes.insert(pos + 1, last.clone_ref(py));

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
        let mut lists = self.get_lists();
        let mut maxes = self.get_maxes();
        lists[pos].remove(idx);
        self.set_len(self.get_len() - 1);

        let len_lists_pos = lists[pos].len();

        if len_lists_pos > (self.get_load() >> 1) {
            maxes[pos] = lists[pos][len_lists_pos - 1].clone_ref(py);

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
            let mut removed = lists[pos]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>();
            lists[prev].append(removed.as_mut());
            maxes[prev] = lists[prev].last().unwrap().clone_ref(py);

            lists.remove(pos);
            maxes.remove(pos);
            self.get_idx().clear();
            drop(maxes);
            self.expand(py, prev)
        } else if len_lists_pos != 0 {
            maxes[pos] = lists[pos][len_lists_pos - 1].clone_ref(py);
            Ok(())
        } else {
            lists.remove(pos);
            maxes.remove(pos);
            self.get_idx().clear();
            Ok(())
        }
    }
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let mut lists = self.get_lists();
        let mut maxes = self.get_maxes();
        if !maxes.is_empty() {
            let mut pos = bisect::right(&maxes, &value.bind(py))?;

            if pos == maxes.len() {
                pos -= 1;
                lists[pos].push(value.clone_ref(py));
                maxes[pos] = value;
            } else {
                let res = bisect::right(&lists[pos], &value.bind(py))?;
                lists[pos].insert(res, value.clone_ref(py));
            }
            drop(maxes);
            self.expand(py, pos)?;
        } else {
            lists.push(vec![value.clone_ref(py)]);
            maxes.push(value);
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

        let pos = bisect::left(&maxes, &value)?;

        if pos == maxes.len() {
            Ok(())
        } else {
            drop(maxes);
            let lists = self.get_lists();
            let idx = bisect::left(&lists[pos], &value)?;

            if lists[pos][idx].bind(py).eq(value)? {
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
            let pos = bisect::left(&maxes, &value)?;

            if pos == maxes.len() {
                errors::not_in_list_err(value)
            } else {
                drop(maxes);
                let lists = self.get_lists();
                let idx = bisect::left(&lists[pos], &value)?;

                if lists[pos][idx].bind(py).eq(&value)? {
                    self.delete(py, pos, idx)
                } else {
                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::left(&maxes, &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let idx = bisect::left(&self.get_lists()[pos], &value)?;
        self.loc(pos, idx as isize)
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::right(&maxes, &value)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let idx = bisect::right(&self.get_lists()[pos], &value)?;
        self.loc(pos, idx as isize)
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos_left = bisect::left(&maxes, &value)?;

        if pos_left == maxes.len() {
            return Ok(0);
        }

        let lists = self.get_lists();
        let idx_left = bisect::left(&lists[pos_left], &value)?;
        let pos_right = bisect::right(&maxes, &value)?;

        if pos_right == maxes.len() {
            return Ok(self.get_len() - self.loc(pos_left, idx_left as isize)? as usize);
        }
        let idx_right = bisect::right(&lists[pos_right], &value)?;

        if pos_left == pos_right {
            return Ok(idx_right - idx_left);
        }

        let right = self.loc(pos_right, idx_right as isize)?;
        let left = self.loc(pos_left, idx_left as isize)?;
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
            let pos_left = bisect::left(&maxes, &value)?;

            if pos_left == maxes.len() {
                errors::is_not_in_list_err(&value)
            } else {
                Ok(pos_left)
            }
        })?;

        let lists = self.get_lists();
        let idx_left = bisect::left(&lists[pos_left], &value)?;

        if lists[pos_left][idx_left].bind(py).ne(&value)? {
            return errors::is_not_in_list_err(&value);
        }

        stop -= 1;
        let left = self.loc(pos_left, idx_left as isize)?;

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
        let mut lists = self.get_lists();
        let mut values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>();

        if !self.get_maxes().is_empty() {
            if values.len() * 4 >= self.get_len() {
                lists.push(values);
                values = self.collapse_lists(py);
                values.sort();
                self.clear();
            } else {
                for val in values {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let values_len = values.len();
        let new_lists = (0..values_len).step_by(load).map(|pos| {
            values[pos..pos + load]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        lists.extend(new_lists);

        let mut new_maxes = lists
            .iter()
            .map(|x| x[x.len() - 1].clone_ref(py))
            .collect::<Vec<_>>();
        self.get_maxes().append(new_maxes.as_mut());
        self.set_len(values_len);
        self.get_idx().clear();
        Ok(())
    }

    fn update_from_vec(&self, py: Python<'_>, mut iterable: Vec<Py<PyAny>>) -> PyResult<()> {
        let mut lists = self.get_lists();
        iterable.sort();

        if !self.get_maxes().is_empty() {
            if iterable.len() * 4 >= self.get_len() {
                lists.push(iterable);
                iterable = self.collapse_lists(py);
                iterable.sort();
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
            iterable[pos..pos + load]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        lists.extend(new_lists);
        self.get_maxes()
            .extend(lists.iter().map(|x| x[x.len() - 1].clone_ref(py)));
        self.set_len(values_len);
        self.get_idx().clear();
        Ok(())
    }
}
