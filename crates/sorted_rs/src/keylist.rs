use crate::{
    bisect,
    bounds::{Indexes, Pos},
    cmp::py_cmp_by_key,
    errors, impl_list_data_getters, ops,
    traits::{DEFAULT_LOAD_FACTOR, ListDataGetters, ListsDataMethods},
};
use crate::Bounds;
use pyo3::prelude::*;
pub struct KeysListsData {
    pub lists: Vec<Vec<Py<PyAny>>>,
    pub keys: Vec<Vec<Py<PyAny>>>,
    pub maxes: Vec<Py<PyAny>>,
    pub idx: Vec<usize>,
    pub len: usize,
    pub offset: usize,
    pub key: Py<PyAny>,
    pub load: usize,
}
impl KeysListsData {
    #[must_use]
    pub fn new(key: Py<PyAny>) -> Self {
        Self {
            lists: Vec::new(),
            keys: Vec::new(),
            maxes: Vec::new(),
            idx: Vec::new(),
            len: 0,
            offset: 0,
            key,
            load: DEFAULT_LOAD_FACTOR,
        }
    }
    pub fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>, key: &Py<PyAny>) -> PyResult<Self> {
        let mut new_inst = Self::new(key.clone_ref(py));
        new_inst.update(py, values)?;
        Ok(new_inst)
    }
}

impl_list_data_getters!(KeysListsData);
impl ListsDataMethods for KeysListsData {
    fn irange_specs<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        let key = self.key.bind(py);
        let minimum = minimum.map(|value| key.call1((value,))).transpose()?;
        let maximum = maximum.map(|value| key.call1((value,))).transpose()?;
        Bounds::get_irange_specs(&self.keys, self.maxes(), minimum, maximum, inclusive)
    }

    fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&self.maxes, &mut bound, &key, bisect::right)? {
            ops::Maxes::Empty => {
                self.lists.push(vec![value]);
                let v = key.unbind();
                self.keys.push(vec![v.clone_ref(py)]);
                self.maxes.push(v);
            }
            ops::Maxes::LenEQPos => {
                bound.pos -= 1;
                let v = key.unbind();
                self.lists[bound.pos].push(value);
                self.keys[bound.pos].push(v.clone_ref(py));
                self.maxes[bound.pos] = v;
                self.expand(py, bound.pos);
            }
            ops::Maxes::LenNEPos => {
                let v = &self.keys[bound.pos];
                bound.idx = bisect::right(v, &key)?;
                self.lists[bound.pos].insert(bound.idx, value);
                self.keys[bound.pos].insert(bound.idx, key.unbind());
                self.expand(py, bound.pos);
            }
        }
        self.len += 1;
        Ok(())
    }

    fn bisect(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: fn(&[pyo3::Py<pyo3::PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>,
    ) -> PyResult<isize> {
        if self.maxes().is_empty() {
            return Ok(0);
        }
        let mut bound = Pos::new(0, 0);

        bound.pos = func(self.maxes(), value)?;

        if bound.pos == self.maxes().len() {
            Ok(self.length().cast_signed())
        } else {
            bound.idx = func(&self.keys[bound.pos], value)?;
            bound.loc(self)
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.lists_mut().clear();
        self.maxes_mut().clear();
        self.idx_mut().clear();
        self.set_len(0);
        self.set_offset(0);
        self.keys.clear();
    }

    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&self.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(false),
            ops::Maxes::LenNEPos => {
                let v = &self.keys[bound.pos];
                bound.idx = bisect::left(v, &key)?;
                let len_keys = self.keys.len();
                let mut len_sublist = self.keys[bound.pos].len();

                loop {
                    if self.keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return Ok(false);
                    }
                    if self.get_value(&bound).bind(py).eq(value)? {
                        return Ok(true);
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(false);
                        }
                        len_sublist = self.keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let mut bound = Pos::default();
        let key = self.key.bind(value.py()).call1((value,))?;
        match ops::Maxes::new(self.maxes(), &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let v_left = &self.keys[bound.pos];
                bound.idx = bisect::left(v_left, &key)?;
                let mut total = 0;
                let len_keys = self.keys.len();
                let mut len_sublist = self.keys[bound.pos].len();
                loop {
                    if self.keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if self.lists()[bound.pos][bound.idx].bind(py).eq(value)? {
                        total += 1;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = self.keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, bounds: &mut Pos) -> PyResult<()> {
        self.keys[bounds.pos].remove(bounds.idx);
        self.lists[bounds.pos].remove(bounds.idx);
        self.len -= 1;
        match ops::Delete::new(&self.keys, self.load, bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.keys[bounds.pos].last().unwrap().clone_ref(py);
                self.delete_on_idx(bounds, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1;
                }
                let prev = bounds.pos - 1;
                let (left, right) = self.keys.split_at_mut(bounds.pos);
                left[prev].append(&mut right[0]);

                let mut removed = self.lists[bounds.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.lists[prev].append(removed.as_mut());
                // NOTE: those three lines below are identical to `remove_pos`, but we have to inline it for the borrow checker to be happy.
                self.lists.remove(bounds.pos);
                self.maxes.remove(bounds.pos);
                self.idx.clear();

                self.maxes[prev] = left[prev].last().unwrap().clone_ref(py);
                self.keys.remove(bounds.pos);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes[bounds.pos] = self.keys[bounds.pos].last().unwrap().clone_ref(py);
            }
            ops::Delete::Other => {
                self.remove_pos(bounds);
                self.keys.remove(bounds.pos);
            }
        }
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&self.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(()),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&self.keys[bound.pos], &key)?;
                let len_keys = self.keys.len();
                let mut len_sublist = self.keys[bound.pos].len();
                loop {
                    if self.keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        break;
                    }
                    if self.get_value(&bound).bind(py).eq(&value)? {
                        self.delete(py, &mut bound)?;
                        break;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            break;
                        }
                        len_sublist = self.keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
                Ok(())
            }
        }
    }

    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self.keys[pos].len(), self.load, &self.idx) {
            ops::Expand::PosLenGtLoad => {
                let half_keys = self.keys[pos].split_off(self.load);
                let half = self.lists[pos].split_off(self.load);
                let new_max_at_pos = self.keys[pos].last().unwrap().clone_ref(py);
                let last_max = half_keys.last().unwrap().clone_ref(py);
                self.expand_at_pos(pos, half, last_max, new_max_at_pos);
                self.keys.insert(pos + 1, half_keys);
            }
            ops::Expand::IdxNotEmpty => self.expand_on_empty_idx(pos),
            ops::Expand::Other => (),
        }
    }

    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let length = self.length().cast_signed();
        if length == 0 {
            errors::not_in_list_err(value)
        } else {
            let mut indexes = Indexes::new(start, stop, length);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(value)
            } else {
                let key = self.key.bind(value.py()).call1((&value,))?;
                let mut bound = Pos {
                    pos: bisect::left(self.maxes(), &key)?,
                    idx: Default::default(),
                };
                if bound.pos == self.maxes().len() {
                    errors::not_in_list_err(value)
                } else {
                    indexes.stop -= 1;
                    let v_left = &self.keys[bound.pos];
                    bound.idx = bisect::left(v_left, &key)?;
                    let len_keys = self.keys.len();
                    let mut len_sublist = v_left.len();

                    loop {
                        if self.keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                            return errors::not_in_list_err(value);
                        }
                        if self.get_value(&bound).bind(py).eq(value)? {
                            let loc = bound.loc(self)?;
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
                                return errors::not_in_list_err(value);
                            }
                            len_sublist = self.keys[bound.pos].len();
                            bound.idx = 0;
                        }
                    }

                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()> {
        let key_fn = &self.key.bind(py);
        let values_len = values.len();
        let new_lists = (0..values_len).step_by(self.load).map(|pos| {
            values[pos..(pos + self.load).min(values_len)]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        self.lists.extend(new_lists);
        let new_keys = self
            .lists
            .iter()
            .map(|list| {
                list.iter()
                    .map(|x| key_fn.call1((x,)).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;

        self.keys.extend(new_keys);
        let new_maxes = self.keys.iter().map(|x| x.last().unwrap().clone_ref(py));
        self.maxes.extend(new_maxes);
        self.len = values_len;
        self.idx.clear();
        Ok(())
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut bound = Pos::default();
        let key = self.key.bind(py).call1((&value,))?;
        match ops::Maxes::new(&self.maxes, &mut bound, &key, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => errors::not_in_list_err(value),
            ops::Maxes::LenNEPos => {
                let v = &self.keys[bound.pos];
                bound.idx = bisect::left(v, &key)?;
                let len_keys = self.keys.len();
                let mut len_sublist = self.keys[bound.pos].len();

                loop {
                    if self.keys[bound.pos][bound.idx].bind(py).ne(&key)? {
                        return errors::not_in_list_err(value);
                    }
                    if self.get_value(&bound).bind(py).eq(value)? {
                        self.delete(py, &mut bound)?;
                        break;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return errors::not_in_list_err(value);
                        }
                        len_sublist = self.keys[bound.pos].len();
                        bound.idx = 0;
                    }
                }
                Ok(())
            }
        }
    }

    fn update(&mut self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        values.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
        match ops::Update::new(&self.maxes, self.len, &values) {
            ops::Update::EmptyMaxes => self.finalize_update(py, &values),
            ops::Update::OtherGESelf => {
                self.lists.push(values);
                values = self.collapse(py);
                values.sort_by(|a, b| py_cmp_by_key(a, b, key_fn));
                self.clear();
                self.finalize_update(py, &values)
            }
            ops::Update::OtherLTSelf => {
                for val in values {
                    self.add(py, val)?;
                }
                Ok(())
            }
        }
    }
}
