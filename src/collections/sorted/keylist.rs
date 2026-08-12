use crate::{
    abc,
    collections::sorted::{
        bisect,
        bounds::{Bounds, Indexes, Pos},
        cmp::py_cmp_by_key,
        data::{ListsData, islice_list, reset_list},
        errors, iter, ops,
        traits::{
            BaseSortedList, BaseSortedListSet, DEFAULT_LOAD_FACTOR, PyIdentity, Reduced,
            SortedCollection, SortedListGetters, try_lock_recover,
        },
    },
    iterators,
    traits::PyoABC,
};
use pyo3::{IntoPyObjectExt, PyTypeInfo, prelude::*, types::PyTuple};
use std::sync::{Mutex, MutexGuard, atomic::AtomicUsize};
use tap::Pipe;
#[pyclass(module = "pyochain.collections", frozen, generic, extends = abc::PyoMutableSequence, sequence)]
pub struct SortedKeyList {
    #[pyo3(get)]
    pub(super) key: Py<PyAny>,
    pub(super) keys: Mutex<Vec<Vec<Py<PyAny>>>>,
    pub(super) data: Mutex<ListsData>,
    pub(super) load: AtomicUsize,
}
impl SortedKeyList {
    pub(super) fn get_keys(&self) -> std::sync::MutexGuard<'_, Vec<Vec<Py<PyAny>>>> {
        try_lock_recover(&self.keys, "keys already locked - reentrant bug")
    }

    pub(super) fn new(key: Py<PyAny>) -> Self {
        Self {
            key,
            keys: Mutex::new(Vec::new()),
            data: Mutex::new(ListsData::default()),
            load: AtomicUsize::new(DEFAULT_LOAD_FACTOR),
        }
    }
    pub(super) fn from_vec<'py>(
        py: Python<'py>,
        values: Vec<Py<PyAny>>,
        key: &Py<PyAny>,
    ) -> PyResult<Self> {
        let new_inst = Self::new(key.clone_ref(py));
        new_inst.update(py, values).map(|_| new_inst)
    }
    pub(super) fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        abc::PyoMutableSequence::build_init()
            .add_subclass(self)
            .pipe(|x| Bound::new(py, x))
    }
}
#[pymethods]
impl SortedKeyList {
    #[pyo3(signature = (min_key = None, max_key = None, inclusive = (true, true), *, reverse = false))]
    pub(super) fn irange_key<'py>(
        slf: Bound<'py, Self>,
        min_key: Option<Bound<'py, PyAny>>,
        max_key: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let slf_ref = slf.get();
        let specs = Bounds::get_irange_specs(
            &slf_ref.get_keys(),
            &slf_ref.get_data().maxes,
            min_key,
            max_key,
            inclusive,
        );
        match specs? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }
    #[new]
    #[pyo3(signature = (iterable = None, *, key = None))]
    fn py_new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        key: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let slf = Self::new(
            key.map(Bound::unbind)
                .unwrap_or_else(|| PyIdentity.into_py_any(py).unwrap()),
        );

        if let Some(iterable) = iterable {
            slf.py_update(&iterable)?;
        }
        abc::PyoMutableSequence::build_init()
            .add_subclass(slf)
            .pipe(Ok)
    }

    pub(super) fn bisect_key_left(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_left(Some(&self.get_keys()), key)
    }
    pub(super) fn bisect_key_right(&self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        self.get_data().bisect_right(Some(&self.get_keys()), &key)
    }
}
impl SortedCollection for SortedKeyList {
    fn __reduce__<'py>(&self, py: Python<'py>) -> Reduced<'py> {
        self.get_data()
            .as_pyovec(py)
            .and_then(|x| PyTuple::new(py, [x.as_any(), &self.key.bind(py)]))
            .map(|tup| (Self::type_object(py), tup))
    }
    fn __contains__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let data = self.get_data();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&data.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(false),
            ops::Maxes::LenNEPos => {
                let keys = self.get_keys();
                let v = &keys[bound.pos];
                bound.idx = bisect::left(&v, &key)?;
                let len_keys = keys.len();
                let mut len_sublist = keys[bound.pos].len();

                loop {
                    if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return Ok(false);
                    }
                    if data.get_value(&bound).bind(py).eq(&value)? {
                        return Ok(true);
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(false);
                        }
                        len_sublist = keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
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
    fn clear(&self, _py: Python<'_>) -> () {
        self.get_data().clear();
        self.get_keys().clear();
    }
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let mut data = self.get_data();
        let length = data.len as isize;
        if length == 0 {
            errors::not_in_list_err(&value)
        } else {
            let mut indexes = Indexes::new(start, stop, length);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(&value)
            } else {
                let key = self.key.bind(py).call1((&value,))?;
                let mut bound = Pos::default();
                bound.pos = bisect::left(&data.maxes, &key)?;
                if bound.pos == data.maxes.len() {
                    errors::not_in_list_err(&value)
                } else {
                    indexes.stop -= 1;
                    let keys = self.get_keys();
                    let v_left = &keys[bound.pos];
                    bound.idx = bisect::left(&v_left, &key)?;
                    let len_keys = keys.len();
                    let mut len_sublist = v_left.len();

                    loop {
                        if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                            return errors::not_in_list_err(&value);
                        }
                        if data.get_value(&bound).bind(py).eq(&value)? {
                            let loc = bound.loc(&mut data)?;
                            if indexes.start <= loc && loc <= indexes.stop {
                                return Ok(loc);
                            } else if loc > indexes.stop {
                                break;
                            }
                        }
                        bound.idx += 1;
                        if bound.idx == len_sublist {
                            bound.pos += 1;
                            if bound.pos == len_keys {
                                return errors::not_in_list_err(&value);
                            }
                            len_sublist = keys[bound.pos].len();
                            bound.idx = 0;
                        }
                    }

                    errors::not_in_list_err(&value)
                }
            }
        }
    }
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let key_fn = |x| slf.get().key.bind(slf.py()).call1((x,));
        let min_key = minimum.map(key_fn).transpose()?;
        let max_key = maximum.map(key_fn).transpose()?;
        Self::irange_key(slf, min_key, max_key, inclusive, reverse)
    }
    fn islice<'py>(
        slf: Bound<'py, Self>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        islice_list(slf, start, stop, reverse)
    }
    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        reset_list(self, py, load)
    }
}
impl BaseSortedListSet for SortedKeyList {
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        let mut data = self.get_data();
        let mut keys = self.get_keys();
        match ops::Maxes::new(&data.maxes, &mut bound, &key, bisect::right)? {
            ops::Maxes::Empty => {
                data.lists.push(vec![value]);
                let v = key.unbind();
                keys.push(vec![v.clone_ref(py)]);
                data.maxes.push(v);
            }
            ops::Maxes::LenEQPos => {
                bound.pos -= 1;
                let v = key.unbind();
                data.lists[bound.pos].push(value);
                keys[bound.pos].push(v.clone_ref(py));
                data.maxes[bound.pos] = v;
                drop(keys);
                self.expand(py, &mut data, bound.pos)?;
            }
            ops::Maxes::LenNEPos => {
                let v = &keys[bound.pos];
                bound.idx = bisect::right(&v, &key)?;
                data.lists[bound.pos].insert(bound.idx, value);
                keys[bound.pos].insert(bound.idx, key.unbind());
                drop(keys);
                self.expand(py, &mut data, bound.pos)?;
            }
        };
        data.len = data.len + 1;
        Ok(())
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&data.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(()),
            ops::Maxes::LenNEPos => {
                let keys = self.get_keys();
                bound.idx = bisect::left(&keys[bound.pos], &key)?;
                let len_keys = keys.len();
                let mut len_sublist = keys[bound.pos].len();
                loop {
                    if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        break;
                    }
                    if data.get_value(&bound).bind(py).eq(&value)? {
                        drop(keys);
                        self.delete(py, &mut data, &mut bound)?;
                        break;
                    } else {
                        bound.idx += 1;
                        if bound.idx == len_sublist {
                            bound.pos += 1;
                            if bound.pos == len_keys {
                                break;
                            } else {
                                len_sublist = keys[bound.pos].len();
                                bound.idx = 0;
                                continue;
                            }
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn remove(&self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut data = self.get_data();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&data.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => errors::not_in_list_err(value),
            ops::Maxes::LenNEPos => {
                let keys = self.get_keys();
                let v = &keys[bound.pos];
                bound.idx = bisect::left(&v, &key)?;
                let len_keys = keys.len();
                let mut len_sublist = keys[bound.pos].len();

                loop {
                    if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return errors::not_in_list_err(value);
                    }
                    if data.get_value(&bound).bind(py).eq(&value)? {
                        drop(keys);
                        self.delete(py, &mut data, &mut bound)?;
                        break;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return errors::not_in_list_err(value);
                        }
                        len_sublist = keys[bound.pos].len();
                        bound.idx = 0
                    }
                }
                Ok(())
            }
        }
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().collapse(py), &self.key)?.into_bound(py)
    }
}
impl BaseSortedList for SortedKeyList {
    fn __add__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let slf_ref = slf.get();
        let out = if other.is(&slf) {
            slf_ref.get_data().repeat(py, 2)
        } else {
            slf_ref.get_data().concat(py, other)?
        };
        Self::from_vec(py, out, &slf_ref.key)?.into_bound(py)
    }

    fn __mul__<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, Self>> {
        Self::from_vec(py, self.get_data().repeat(py, num), &self.key)?.into_bound(py)
    }

    //recursive_repr()
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let type_name = Self::type_object(py).name()?;
        let key_repr = self.key.bind(py).repr()?;

        self.get_data()
            .py_repr(py)
            .map(|repr| format!("{type_name}({}, key={})", repr, key_repr))
    }

    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        iter::SortedIterKey::new(py, inner)
    }
    fn delete(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        bounds: &mut Pos,
    ) -> PyResult<()> {
        let mut keys = self.get_keys();

        keys[bounds.pos].remove(bounds.idx);
        data.lists[bounds.pos].remove(bounds.idx);
        data.len = data.len - 1;
        match ops::Delete::new(&keys, self.get_load(), bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = keys[bounds.pos].last().unwrap().clone_ref(py);
                data.delete_on_idx(bounds, max_at_pos)
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1
                }
                let prev = bounds.pos - 1;
                let (left, right) = keys.split_at_mut(bounds.pos);
                left[prev].extend(right[0].drain(..));
                data.set_prev_from_removed(py, bounds, prev);
                data.maxes[prev] = left[prev].last().unwrap().clone_ref(py);
                keys.remove(bounds.pos);
                drop(keys);
                self.expand(py, data, prev)?
            }
            ops::Delete::LenPosNotZero => {
                data.maxes[bounds.pos] = keys[bounds.pos].last().unwrap().clone_ref(py)
            }
            ops::Delete::Other => {
                data.remove_pos(bounds);
                keys.remove(bounds.pos);
            }
        };
        Ok(())
    }
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()> {
        let load = self.get_load();
        let mut keys = self.get_keys();
        match ops::Expand::new(keys[pos].len(), load, &data.idx) {
            ops::Expand::PosLenGtLoad => {
                let half_keys = keys[pos].split_off(load);
                let half = data.lists[pos].split_off(load);
                let new_max_at_pos = keys[pos].last().unwrap().clone_ref(py);
                let last_max = half_keys.last().unwrap().clone_ref(py);
                data.expand_at_pos(pos, half, last_max, new_max_at_pos);
                keys.insert(pos + 1, half_keys);
                Ok(())
            }
            ops::Expand::IdxNotEmpty => data.expand_on_empty_idx(pos),
            ops::Expand::Other => Ok(()),
        }
    }
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let data = self.get_data();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&data.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let keys = self.get_keys();
                let v_left = &keys[bound.pos];
                bound.idx = bisect::left(&v_left, &key)?;
                let mut total = 0;
                let len_keys = keys.len();
                let mut len_sublist = keys[bound.pos].len();
                loop {
                    if keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if data.lists[bound.pos][bound.idx].bind(py).eq(&value)? {
                        total += 1;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }
    fn update(&self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
        let mut data = self.get_data();
        let mut keys = self.get_keys();
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        values.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
        let load = self.get_load();
        match ops::Update::new(&data.maxes, data.len, &values) {
            ops::Update::EmptyMaxes => {
                finalize_update(&mut data, py, values, load, &mut keys, key_fn)
            }
            ops::Update::OtherGESelf => {
                data.lists.push(values);
                values = data.collapse(py);
                values.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
                data.clear();
                keys.clear();
                finalize_update(&mut data, py, values, load, &mut keys, key_fn)
            }
            ops::Update::OtherLTSelf => {
                drop(data);
                drop(keys);
                for val in values {
                    self.add(py, val)?;
                }
                Ok(())
            }
        }
    }
}
fn finalize_update(
    data: &mut MutexGuard<'_, ListsData>,
    py: Python<'_>,
    values: Vec<Py<PyAny>>,
    load: usize,
    keys: &mut MutexGuard<'_, Vec<Vec<Py<PyAny>>>>,
    key_fn: &Bound<'_, PyAny>,
) -> PyResult<()> {
    let values_len = values.len();
    let new_lists = (0..values_len).step_by(load).map(|pos| {
        values[pos..(pos + load).min(values_len)]
            .iter()
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>()
    });
    data.lists.extend(new_lists);
    let new_keys = data
        .lists
        .iter()
        .map(|list| {
            list.iter()
                .map(|x| key_fn.call1((x,)).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()
        })
        .collect::<PyResult<Vec<_>>>()?;

    keys.extend(new_keys);
    let new_maxes = keys.iter().map(|x| x.last().unwrap().clone_ref(py));
    data.maxes.extend(new_maxes);
    data.len = values_len;
    data.idx.clear();
    Ok(())
}
