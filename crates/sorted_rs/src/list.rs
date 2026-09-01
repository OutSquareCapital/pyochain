use pyo3::prelude::*;

use crate::{
    bisect,
    bounds::{Bounds, Indexes, Pos},
    cmp::py_cmp,
    errors, impl_list_data_getters, ops,
    traits::{DEFAULT_LOAD_FACTOR, ListDataGetters, ListsDataMethods},
};

//TODO: This struct is way too big and do way too many things.
// Unfortunately we must first decouple as much as possible code from the main src/ folder into this crate.
pub struct ListsData {
    pub lists: Vec<Vec<Py<PyAny>>>,
    pub maxes: Vec<Py<PyAny>>,
    pub idx: Vec<usize>,
    pub len: usize,
    pub offset: usize,
    pub load: usize,
}
impl Default for ListsData {
    fn default() -> Self {
        Self {
            lists: Vec::new(),
            maxes: Vec::new(),
            idx: Vec::new(),
            len: 0,
            offset: 0,
            load: DEFAULT_LOAD_FACTOR,
        }
    }
}
impl ListsData {
    pub fn from_vec(py: Python<'_>, values: Vec<Py<PyAny>>) -> PyResult<Self> {
        let mut new_inst = Self::default();
        new_inst.update(py, values)?;
        Ok(new_inst)
    }
}
impl_list_data_getters!(ListsData);
impl ListsDataMethods for ListsData {
    fn irange_specs<'py>(
        &self,
        _py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        Bounds::from_sorted(self.lists(), self.maxes(), minimum, maximum, inclusive)
    }

    fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let mut bound = Pos::default();
        match ops::Maxes::new(&self.maxes, &mut bound, value.bind(py), bisect::right)? {
            ops::Maxes::Empty => {
                self.lists.push(vec![value.clone_ref(py)]);
                self.maxes.push(value);
            }
            ops::Maxes::LenEQPos => {
                bound.pos -= 1;
                self.lists[bound.pos].push(value.clone_ref(py));
                self.maxes[bound.pos] = value;
                self.expand(py, bound.pos);
            }
            ops::Maxes::LenNEPos => {
                let res = bisect::right(&self.lists[bound.pos], value.bind(py))?;
                self.lists[bound.pos].insert(res, value.clone_ref(py));
                self.expand(py, bound.pos);
            }
        }
        self.len += 1;
        Ok(())
    }
    #[inline]
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
            bound.idx = func(&self.lists()[bound.pos], value)?;
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
    }
    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let mut bound = Pos::default();
        match ops::Maxes::new(&self.maxes, &mut bound, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(false),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&self.lists[bound.pos], value)?;
                self.get_value(&bound).bind(py).eq(value)
            }
        }
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut left = Pos::default();
        match ops::Maxes::new(self.maxes(), &mut left, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(0),
            ops::Maxes::LenNEPos => {
                let mut right = Pos::default();
                left.idx = bisect::left(&self.lists()[left.pos], value)?;
                right.pos = bisect::right(self.maxes(), value)?;

                if right.pos == self.maxes().len() {
                    let left_loc = left.loc(self)?;
                    Ok(self.length() - left_loc.cast_unsigned())
                } else {
                    right.idx = bisect::right(&self.lists()[right.pos], value)?;

                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_loc = right.loc(self)?;
                        let left_loc = left.loc(self)?;
                        Ok((right_loc - left_loc).cast_unsigned())
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, bounds: &mut Pos) -> PyResult<()> {
        self.lists[bounds.pos].remove(bounds.idx);
        self.len -= 1;
        match ops::Delete::new(&self.lists, self.load, bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.lists[bounds.pos].last().unwrap().clone_ref(py);
                self.delete_on_idx(bounds, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1;
                }
                let prev = bounds.pos - 1;
                let mut removed = self.lists()[bounds.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.lists_mut()[prev].append(removed.as_mut());
                self.remove_pos(bounds);
                self.maxes[prev] = self.lists[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes[bounds.pos] = self.lists[bounds.pos].last().unwrap().clone_ref(py);
            }
            ops::Delete::Other => self.remove_pos(bounds),
        }
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut bound = Pos::default();
        match ops::Maxes::new(&self.maxes, &mut bound, &value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => Ok(()),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&self.lists[bound.pos], &value)?;
                if self.get_value(&bound).bind(py).eq(&value)? {
                    self.delete(py, &mut bound)
                } else {
                    Ok(())
                }
            }
        }
    }

    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self.lists[pos].len(), self.load, &self.idx) {
            ops::Expand::PosLenGtLoad => {
                let half = self.lists[pos].split_off(self.load);
                let new_max_at_pos = self.lists[pos].last().unwrap().clone_ref(py);
                let last_max = half.last().unwrap().clone_ref(py);
                self.expand_at_pos(pos, half, last_max, new_max_at_pos);
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
        let len_ = self.length().cast_signed();

        if len_ == 0 {
            errors::not_in_list_err(value)
        } else {
            let mut indexes = Indexes::new(start, stop, len_);
            if indexes.stop <= indexes.start {
                errors::not_in_list_err(value)
            } else {
                let mut bound = Pos {
                    pos: bisect::left(self.maxes(), value)?,
                    idx: 0,
                };
                if bound.pos == self.maxes().len() {
                    errors::not_in_list_err(value)
                } else {
                    bound.idx = bisect::left(&self.lists()[bound.pos], value)?;
                    if self.get_value(&bound).bind(py).ne(value)? {
                        errors::not_in_list_err(value)
                    } else {
                        indexes.stop -= 1;
                        let left = bound.loc(self)?;

                        if indexes.start <= left {
                            if left <= indexes.stop {
                                return Ok(left);
                            }
                        } else {
                            let right = self.bisect_right(value)? - 1;

                            if indexes.start <= right {
                                return Ok(indexes.start);
                            }
                        }
                        errors::not_in_list_err(value)
                    }
                }
            }
        }
    }

    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let mut bound = Pos::default();
        match ops::Maxes::new(&self.maxes, &mut bound, value, bisect::left)? {
            ops::Maxes::Empty | ops::Maxes::LenEQPos => errors::not_in_list_err(value),
            ops::Maxes::LenNEPos => {
                bound.idx = bisect::left(&self.lists[bound.pos], value)?;
                if self.get_value(&bound).bind(py).eq(value)? {
                    self.delete(py, &mut bound)
                } else {
                    errors::not_in_list_err(value)
                }
            }
        }
    }

    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()> {
        let values_len = values.len();
        let new_lists = (0..values_len).step_by(self.load).map(|pos| {
            values[pos..(pos + self.load).min(values_len)]
                .iter()
                .map(|x| x.clone_ref(py))
                .collect::<Vec<_>>()
        });
        self.lists.extend(new_lists);
        let mut new_maxes = self
            .lists
            .iter()
            .map(|x| x.last().unwrap().clone_ref(py))
            .collect::<Vec<_>>();
        self.maxes.append(new_maxes.as_mut());
        self.len = values_len;
        self.idx.clear();
        Ok(())
    }
    fn update(&mut self, py: Python<'_>, mut values: Vec<Py<PyAny>>) -> PyResult<()> {
        values.sort_by(|a, b| py_cmp(py, a, b));
        match ops::Update::new(self.maxes(), self.len, &values) {
            ops::Update::EmptyMaxes => self.finalize_update(py, &values),
            ops::Update::OtherGESelf => {
                self.lists.push(values);
                values = self.collapse(py);
                values.sort_by(|a, b| py_cmp(py, a, b));
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
