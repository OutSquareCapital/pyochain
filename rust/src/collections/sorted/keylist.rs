use super::errors;
use crate::collections::sorted::bisect;
use crate::collections::sorted::iter::try_iterator_into_list;
use crate::collections::sorted::traits::{
    DEFAULT_LOAD_FACTOR, InnerSorted, InnerSortedGetters, RustGetters,
};
use crate::pyo3_ext::{prelude::*, pylibs};
use crate::seq::{IntoPyochain, PyoVec};
use pyo3::{prelude::*, types::PyList};
use std::ops::Index;
use std::sync::{Mutex, atomic::AtomicUsize};
use tap::Pipe;
#[pyclass(generic, frozen)]
pub struct InnerKeyLists {
    #[pyo3(get)]
    pub(super) key: Py<PyAny>,
    #[pyo3(get)]
    pub(super) keys: Py<PyoVec>,
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
impl InnerKeyLists {
    #[new]
    fn new(key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = key.py();
        Ok(Self {
            key: key.unbind(),
            keys: PyoVec::new_bound(py)?.unbind(),
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: Mutex::new(Vec::new()),
            len: AtomicUsize::new(0),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
            offset: AtomicUsize::new(0),
        })
    }
}
impl InnerSorted for InnerKeyLists {
    fn clear(&self, py: Python<'_>) -> () {
        self.set_len(0);
        self.lists.get().clear(py);
        self.keys.get().clear(py);
        self.maxes.get().clear(py);
        self.get_idx().clear();
    }
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let maxes = self.maxes.bind(py);

        if maxes.is_empty()? {
            return Ok(false);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(maxes, &key)?;

        if pos == maxes.len()? {
            return Ok(false);
        }

        let lists = self.lists.bind(py);
        let keys = self.keys.bind(py);
        let v = &keys
            .get_item(&pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v, &key)?;

        let len_keys = keys.len()?;
        let mut len_sublist = keys.get_item(&pos)?.len()?;

        loop {
            if keys.get_item(&pos)?.get_item(&idx)?.ne(&key)? {
                return Ok(false);
            }
            if lists.get_item(&pos)?.get_item(&idx)?.eq(&value)? {
                return Ok(true);
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(false);
                }
                len_sublist = keys.get_item(&pos)?.len()?;
                idx = 0;
            }
        }
    }
    fn delete(&self, py: Python<'_>, mut pos: usize, idx: usize) -> PyResult<()> {
        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        let keys_pos = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);
        let lists_pos = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);

        keys_pos.del_item(idx)?;
        lists_pos.del_item(idx)?;
        self.set_len(self.get_len() - 1);

        let len_keys_pos = keys_pos.len();

        if len_keys_pos > (self.get_load() >> 1) {
            maxes.set_item(pos, keys_pos.last()?)?;

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
            keys.get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(keys.get_item(pos)?)?;
            lists
                .get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(lists.get_item(pos)?)?;
            maxes.set_item(prev, keys.get_item(prev)?.get_item(-1)?)?;

            lists.del_item(pos)?;
            keys.del_item(pos)?;
            maxes.del_item(pos)?;
            self.get_idx().clear();

            self.expand(py, prev)
        } else if len_keys_pos != 0 {
            maxes.set_item(pos, keys_pos.last()?)
        } else {
            lists.del_item(pos)?;
            keys.del_item(pos)?;
            maxes.del_item(pos)?;
            self.get_idx().clear();
            Ok(())
        }
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);

        if keys.get_item(pos)?.len()? > self.get_load() << 1 {
            let maxes = self.maxes.get().inner.bind(py);
            let load = self.get_load();

            let lists_pos = lists
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let keys_pos = keys
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let half = lists_pos.get_slice(load, lists_pos.len()).into_pyochain()?;
            let half_keys = keys_pos.get_slice(load, keys_pos.len()).into_pyochain()?;
            lists_pos.del_slice(load, usize::MAX)?;
            keys_pos.del_slice(load, usize::MAX)?;
            maxes.set_item(pos, keys_pos.last()?)?;

            lists.insert(pos + 1, half)?;
            keys.insert(pos + 1, &half_keys)?;
            maxes.insert(pos + 1, half_keys.get_item(half_keys.len()? - 1)?)?;
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
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let key = self.key.bind(py).call1((&value,))?;
        let lists = self.lists.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);

        if !maxes.is_empty() {
            let mut pos = bisect::bisect_right(self.maxes.bind(py), &key)?;

            if pos == maxes.len() {
                pos -= 1;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&value)?;
                keys.get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&key)?;
                maxes.set_item(pos, &key)?;
            } else {
                let v = &keys
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
                let idx = bisect::bisect_right(&v, &key)?;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .insert(idx, &value)?;
                keys.get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .insert(idx, &key)?;
            }

            self.expand(py, pos)?;
        } else {
            lists.append(PyList::new(py, [value])?.into_pyochain()?)?;
            keys.append(PyList::new(py, [&key])?.into_pyochain()?)?;
            maxes.append(key)?;
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

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(());
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v = &keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                break;
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                self.delete(py, pos, idx)?;
                break;
            } else {
                idx += 1;
                if idx == len_sublist {
                    pos += 1;
                    if pos == len_keys {
                        break;
                    } else {
                        len_sublist = keys.get_item(pos)?.len()?;
                        idx = 0;
                        continue;
                    }
                }
            }
        }
        Ok(())
    }

    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return errors::not_in_list_err(value);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return errors::not_in_list_err(value);
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v = &keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

        let mut idx = bisect::bisect_left(&v, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return errors::not_in_list_err(value);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                self.delete(py, pos, idx)?;
                break;
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return errors::not_in_list_err(value);
                }
                len_sublist = keys.get_item(pos)?.len()?;
                idx = 0
            }
        }
        Ok(())
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
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(0);
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v_left = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v_left, &key)?;
        let mut total = 0;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return Ok(total);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                total += 1;
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(total);
                }
                len_sublist = keys.get_item(pos)?.len()?;
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
        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return errors::is_not_in_list_err(value);
        }

        stop -= 1;
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);
        let v_left = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v_left, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = v_left.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return errors::is_not_in_list_err(value);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                let loc = self.loc(py, pos, idx as isize)?;
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
                    return errors::is_not_in_list_err(value);
                }
                len_sublist = keys.get_item(pos)?.len()?;
                idx = 0;
            }
        }

        errors::is_not_in_list_err(value)
    }
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);
        let maxes = self.maxes.get().inner.clone_ref(py).into_bound(py);
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|x| pylibs::builtins::sorted_by(&x, false, key_fn))?;

        if !maxes.is_empty() {
            if values.len() * 4 >= self.get_len() {
                lists.append(values.into_pyochain()?)?;
                values = self
                    .collapse_lists(py)?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py);
                values.sort_by(key_fn, false)?;
                self.clear(py);
            } else {
                for val in values {
                    self.add(val)?;
                }
                return Ok(());
            }
        }

        let load = self.get_load();
        let new_maxes = (0..values.len())
            .step_by(load)
            .map(|pos| values.get_slice(pos, pos + load).into_pyochain())
            .try_fold(lists, try_iterator_into_list)?
            .iter()
            .map(|list_| {
                unsafe { list_.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .iter()
                    .map(|x| key_fn.call1((x,)))
                    .try_fold(PyList::empty(py), |acc, x| {
                        acc.append(x?)?;
                        Ok::<_, PyErr>(acc)
                    })?
                    .into_pyochain()
            })
            .try_fold(keys, try_iterator_into_list)?
            .iter()
            .map(|x| {
                unsafe { x.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .last()
            })
            .try_fold(PyList::empty(py), try_iterator_into_list)?;
        maxes.iadd(&new_maxes)?;
        self.set_len(values.len());
        self.get_idx().clear();
        Ok(())
    }
}

#[pymethods]
impl InnerKeyLists {
    fn bisect_key_left(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = key.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            Ok(self.get_len() as isize)
        } else {
            let v = self
                .keys
                .bind(py)
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
            let idx = bisect::bisect_left(&v, &key)?;

            self.loc(py, pos, idx as isize)
        }
    }
    fn bisect_key_right(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = key.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_right(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(self.get_len() as isize);
        }
        let v = &self
            .keys
            .get()
            .inner
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_right(&v, &key)?;

        return self.loc(py, pos, idx as isize);
    }
}
