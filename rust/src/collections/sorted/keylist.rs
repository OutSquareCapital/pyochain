use super::errors;
use crate::collections::sorted::bisect;
use crate::collections::sorted::cmp::py_cmp_by_key;
use crate::collections::sorted::traits::{
    DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters,
};
use crate::pyo3_ext::pylibs;
use pyo3::prelude::*;
use std::ops::Index;
use std::sync::{Mutex, atomic::AtomicUsize};
use tap::Pipe;
#[pyclass(module = "pyochain._collections", frozen, generic)]
pub struct InnerKeyLists {
    #[pyo3(get)]
    pub(super) key: Py<PyAny>,
    pub(super) keys: Mutex<Vec<Vec<Py<PyAny>>>>,
    pub(super) lists: Mutex<Vec<Vec<Py<PyAny>>>>,
    pub(super) maxes: Mutex<Vec<Py<PyAny>>>,
    pub(super) idx: Mutex<Vec<usize>>,
    pub(super) len: AtomicUsize,
    pub(super) load: AtomicUsize,
    pub(super) offset: AtomicUsize,
}
impl InnerKeyLists {
    pub(super) fn get_keys(&self) -> std::sync::MutexGuard<'_, Vec<Vec<Py<PyAny>>>> {
        self.keys
            .try_lock()
            .expect("keys already locked - reentrant bug")
    }
}
#[pymethods]
impl InnerKeyLists {
    #[new]
    fn new(key: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            key: key.unbind(),
            keys: Mutex::new(Vec::new()),
            lists: Mutex::new(Vec::new()),
            maxes: Mutex::new(Vec::new()),
            idx: Mutex::new(Vec::new()),
            len: AtomicUsize::new(0),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
            offset: AtomicUsize::new(0),
        })
    }
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true)))]
    fn irange_key(
        &self,
        min_key: Option<Bound<'_, PyAny>>,
        max_key: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<(usize, usize, usize, usize)>> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(None);
        }

        let keys = self.get_keys();

        // Calculate the minimum (pos, idx) pair. By default this location
        // will be inclusive in our calculation.
        let (min_pos, min_idx) = match min_key {
            None => (0, 0),
            Some(min_key) => {
                if inclusive.0 {
                    let min_pos = bisect::left(&maxes, &min_key)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::left(&keys[min_pos], &min_key)?;
                    (min_pos, min_idx)
                } else {
                    let min_pos = bisect::right(&maxes, &min_key)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::right(&keys[min_pos], &min_key)?;
                    (min_pos, min_idx)
                }
            }
        };

        // Calculate the maximum (pos, idx) pair. By default this location
        // will be exclusive in our calculation.
        let (max_pos, max_idx) = match max_key {
            None => {
                let max_pos = maxes.len() - 1;
                let max_idx = keys[max_pos].len();
                (max_pos, max_idx)
            }

            Some(max_key) => {
                if inclusive.1 {
                    let mut max_pos = bisect::right(&maxes, &max_key)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        keys[max_pos].len()
                    } else {
                        bisect::right(&keys[max_pos], &max_key)?
                    };
                    (max_pos, max_idx)
                } else {
                    let mut max_pos = bisect::left(&maxes, &max_key)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        keys[max_pos].len()
                    } else {
                        bisect::left(&keys[max_pos], &max_key)?
                    };
                    (max_pos, max_idx)
                }
            }
        };
        Ok(Some((min_pos, min_idx, max_pos, max_idx)))
    }
}
impl InnerSorted for InnerKeyLists {
    fn clear(&self) -> () {
        self.set_len(0);
        self.get_lists().clear();
        self.get_keys().clear();
        self.get_maxes().clear();
        self.get_idx().clear();
    }
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(false);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            return Ok(false);
        }

        let lists = self.get_lists();
        let keys = self.get_keys();
        let v = &keys[pos];
        let mut idx = bisect::left(&v, &key)?;

        let len_keys = keys.len();
        let mut len_sublist = keys[pos].len();

        loop {
            if keys[pos][idx].bind(py).ne(&key)? {
                return Ok(false);
            }
            if lists[pos][idx].bind(py).eq(&value)? {
                return Ok(true);
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(false);
                }
                len_sublist = keys[pos].len();
                idx = 0;
            }
        }
    }
    fn delete(&self, py: Python<'_>, mut pos: usize, idx: usize) -> PyResult<()> {
        let mut lists = self.get_lists();
        let mut keys = self.get_keys();
        let mut maxes = self.get_maxes();

        keys[pos].remove(idx);
        lists[pos].remove(idx);
        self.set_len(self.get_len() - 1);

        let len_keys_pos = keys[pos].len();

        if len_keys_pos > (self.get_load() >> 1) {
            maxes[pos] = keys[pos][len_keys_pos - 1].clone_ref(py);

            self.get_idx().pipe_ref_mut(|index| {
                if !index.is_empty() {
                    let mut child = self.get_offset() + pos;
                    while child > 0 {
                        index[child] = index[child] - 1;
                        child = (child - 1) >> 1;
                    }
                    index[0] = index[0] - 1;
                }
            });
            Ok(())
        } else if keys.len() > 1 {
            if pos == 0 {
                pos += 1
            }

            let prev = pos - 1;
            let (left, right) = keys.split_at_mut(pos);
            left[prev].extend(right[0].drain(..));
            let mut removed = lists[pos]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>();
            lists[prev].append(removed.as_mut());
            maxes[prev] = keys[prev][keys[prev].len() - 1].clone_ref(py);

            lists.remove(pos);
            keys.remove(pos);
            maxes.remove(pos);
            self.get_idx().clear();
            drop(maxes);
            drop(keys);

            self.expand(py, prev)
        } else if len_keys_pos != 0 {
            maxes[pos] = keys[pos][len_keys_pos - 1].clone_ref(py);
            Ok(())
        } else {
            lists.remove(pos);
            keys.remove(pos);
            maxes.remove(pos);
            self.get_idx().clear();
            Ok(())
        }
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let mut lists = self.get_lists();
        let mut keys = self.get_keys();

        if keys[pos].len() > self.get_load() << 1 {
            let mut maxes = self.get_maxes();
            let load = self.get_load();
            let keys_pos = &mut keys[pos];
            let half_keys = keys_pos.split_off(load);
            let half = lists[pos].split_off(load);
            maxes[pos] = keys_pos[keys_pos.len() - 1].clone_ref(py);

            lists.insert(pos + 1, half);
            maxes.insert(pos + 1, half_keys[half_keys.len() - 1].clone_ref(py));
            keys.insert(pos + 1, half_keys);
            self.get_idx().clear();
            Ok(())
        } else if !self.get_idx().is_empty() {
            self.get_idx().pipe_ref_mut(|index| {
                let mut child = self.get_offset() + pos;
                while child != 0 {
                    index[child] = index.index(child) + &1;
                    child = (child - 1) >> 1;
                }
                index[0] = index.index(0) + &1;
            });
            Ok(())
        } else {
            Ok(())
        }
    }
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let key = self.key.bind(py).call1((&value,))?.unbind();
        let key_binded = key.bind(py);
        let mut lists = self.get_lists();
        let mut maxes = self.get_maxes();
        let mut keys = self.get_keys();

        if !maxes.is_empty() {
            let mut pos = bisect::right(&maxes, &key_binded)?;

            if pos == maxes.len() {
                pos -= 1;
                lists[pos].push(value);
                keys[pos].push(key.clone_ref(py));
                maxes[pos] = key;
            } else {
                let v = &keys[pos];
                let idx = bisect::right(&v, &key_binded)?;
                lists[pos].insert(idx, value);
                keys[pos].insert(idx, key);
            }
            drop(maxes);
            drop(keys);
            self.expand(py, pos)?;
        } else {
            lists.push(vec![value]);
            keys.push(vec![key.clone_ref(py)]);
            maxes.push(key);
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

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            Ok(())
        } else {
            drop(maxes);
            let lists = self.get_lists();
            let keys = self.get_keys();

            let mut idx = bisect::left(&keys[pos], &key)?;
            let len_keys = keys.len();
            let mut len_sublist = keys[pos].len();

            loop {
                if keys[pos][idx].bind(py).ne(&key)? {
                    break;
                }
                if lists[pos][idx].bind(py).eq(&value)? {
                    drop(keys);
                    self.delete(py, pos, idx)?;
                    break;
                } else {
                    idx += 1;
                    if idx == len_sublist {
                        pos += 1;
                        if pos == len_keys {
                            break;
                        } else {
                            len_sublist = keys[pos].len();
                            idx = 0;
                            continue;
                        }
                    }
                }
            }
            Ok(())
        }
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return errors::not_in_list_err(value);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            errors::not_in_list_err(value)
        } else {
            drop(maxes);
            let lists = self.get_lists();
            let keys = self.get_keys();
            let v = &keys[pos];

            let mut idx = bisect::left(&v, &key)?;
            let len_keys = keys.len();
            let mut len_sublist = keys[pos].len();

            loop {
                if keys[pos][idx].bind(py).ne(&key)? {
                    return errors::not_in_list_err(value);
                }
                if lists[pos][idx].bind(py).eq(&value)? {
                    drop(keys);
                    self.delete(py, pos, idx)?;
                    break;
                }
                idx += 1;
                if idx == len_sublist {
                    pos += 1;
                    if pos == len_keys {
                        return errors::not_in_list_err(value);
                    }
                    len_sublist = keys[pos].len();
                    idx = 0
                }
            }
            Ok(())
        }
    }
    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.key
            .bind(value.py())
            .call1((value,))
            .and_then(|x| self.bisect_key_left(x))
    }

    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.key
            .bind(value.py())
            .call1((value,))
            .and_then(|x| self.bisect_key_right(x))
    }

    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            return Ok(0);
        }

        let lists = self.get_lists();
        let keys = self.get_keys();
        let v_left = &keys[pos];
        let mut idx = bisect::left(&v_left, &key)?;
        let mut total = 0;
        let len_keys = keys.len();
        let mut len_sublist = keys[pos].len();

        loop {
            if keys[pos][idx].bind(py).ne(&key)? {
                return Ok(total);
            }
            if lists[pos][idx].bind(py).eq(&value)? {
                total += 1;
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(total);
                }
                len_sublist = keys[pos].len();
                idx = 0;
            }
        }
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

        let maxes = self.get_maxes();
        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            return errors::is_not_in_list_err(&value);
        }

        stop -= 1;
        let lists = self.get_lists();
        let keys = self.get_keys();
        let v_left = &keys[pos];
        let mut idx = bisect::left(&v_left, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = v_left.len();

        loop {
            if keys[pos][idx].bind(py).ne(&key)? {
                return errors::is_not_in_list_err(&value);
            }
            if lists[pos][idx].bind(py).eq(&value)? {
                let loc = self.loc(pos, idx as isize)?;
                if start <= loc && loc <= stop {
                    return Ok(loc);
                } else if loc > stop {
                    break;
                }
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return errors::is_not_in_list_err(&value);
                }
                len_sublist = keys[pos].len();
                idx = 0;
            }
        }

        errors::is_not_in_list_err(&value)
    }
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let mut lists = self.get_lists();
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|x| pylibs::builtins::sorted_by(&x, false, key_fn))?
            .iter()
            .map(Bound::unbind)
            .collect::<Vec<_>>();

        if !self.get_maxes().is_empty() {
            if values.len() * 4 >= self.get_len() {
                lists.push(values);
                values = self.collapse_lists(py);
                values.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
                self.clear();
            } else {
                for val in values {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let new_lists = (0..values.len()).step_by(load).map(|pos| {
            values[pos..pos + load]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        lists.extend(new_lists);
        let new_keys = lists
            .iter()
            .map(|list| {
                list.iter()
                    .map(|x| key_fn.call1((x,)).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut keys = self.get_keys();
        keys.extend(new_keys);
        let new_maxes = keys.iter().map(|x| x[x.len() - 1].clone_ref(py));
        self.get_maxes().extend(new_maxes);
        self.set_len(values.len());
        self.get_idx().clear();
        Ok(())
    }
    fn update_from_vec(&self, py: Python<'_>, mut iterable: Vec<Py<PyAny>>) -> PyResult<()> {
        let mut lists = self.get_lists();
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        iterable.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));

        if !self.get_maxes().is_empty() {
            if iterable.len() * 4 >= self.get_len() {
                lists.push(iterable);
                iterable = self.collapse_lists(py);
                iterable.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
                self.clear();
            } else {
                for val in iterable {
                    self.add(py, val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let new_lists = (0..iterable.len()).step_by(load).map(|pos| {
            iterable[pos..pos + load]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        lists.extend(new_lists);
        let new_keys = lists
            .iter()
            .map(|list| {
                list.iter()
                    .map(|x| key_fn.call1((x,)).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        let mut keys = self.get_keys();
        keys.extend(new_keys);
        let new_maxes = keys.iter().map(|x| x[x.len() - 1].clone_ref(py));
        self.get_maxes().extend(new_maxes);
        self.set_len(iterable.len());
        self.get_idx().clear();
        Ok(())
    }
}

#[pymethods]
impl InnerKeyLists {
    fn bisect_key_left(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::left(&maxes, &key)?;

        if pos == maxes.len() {
            Ok(self.get_len() as isize)
        } else {
            let idx = bisect::left(&self.get_keys()[pos], &key)?;

            self.loc(pos, idx as isize)
        }
    }
    fn bisect_key_right(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let maxes = self.get_maxes();

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::right(&maxes, &key)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let idx = bisect::right(&self.get_keys()[pos], &key)?;

        return self.loc(pos, idx as isize);
    }
}
