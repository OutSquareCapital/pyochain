use crate::{
    Bounds,
    bisect::Bisect,
    bounds::Pos,
    cmp::py_cmp_by_key,
    errors, impl_inner_getter,
    inner::{InnerData, InnerGetter, ListDataGetters, VecPy},
    ops,
    traits::{ListsDataMethods, NestedVec, update_list_by},
};
use pyo3::prelude::*;
use tap::prelude::*;
pub struct KeysListsData(InnerData, pub Vec<VecPy>, pub Py<PyAny>);

impl KeysListsData {
    #[must_use]
    pub fn new(key: Py<PyAny>) -> Self {
        Self(InnerData::default(), Vec::default(), key)
    }
    pub fn from_vec(py: Python<'_>, values: VecPy, key: Py<PyAny>) -> PyResult<Self> {
        let mut new_inst = Self::new(key);
        new_inst.update(py, values)?;
        Ok(new_inst)
    }
}
impl_inner_getter!(KeysListsData);
impl ListsDataMethods for KeysListsData {
    fn irange_specs<'py>(
        &self,
        py: Python<'py>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        let key = self.2.bind(py);
        let minimum = minimum.map(|value| key.call1((value,))).transpose()?;
        let maximum = maximum.map(|value| key.call1((value,))).transpose()?;
        Bounds::from_sorted(&self.1, self.maxes(), minimum, maximum, inclusive)
    }

    fn add(&mut self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()> {
        let key = self.2.bind(py).call1((&value,))?;
        match ops::Maxes::right(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                self.lists_mut().push(vec![value]);
                let v = key.unbind();
                self.1.push(vec![v.clone_ref(py)]);
                self.maxes_mut().push(v);
            }
            ops::Maxes::LenEQPos(mut bound) => {
                bound.pos -= 1;
                let v = key.unbind();
                self.lists_mut()[bound.pos].push(value);
                self.1[bound.pos].push(v.clone_ref(py));
                self.maxes_mut()[bound.pos] = v;
                self.expand(py, bound.pos);
            }
            ops::Maxes::LenNEPos(mut bound) => {
                let v = &self.1[bound.pos];
                bound.idx = v.bisect_right(&key)?;
                self.lists_mut()[bound.pos].insert(bound.idx, value);
                self.1[bound.pos].insert(bound.idx, key.unbind());
                self.expand(py, bound.pos);
            }
        }
        self.increment_len();
        Ok(())
    }

    fn bisect(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: fn(&[pyo3::Py<pyo3::PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>,
    ) -> PyResult<isize> {
        if self.maxes().is_empty() {
            Ok(0)
        } else {
            let mut bound = Pos::new(0, 0);
            bound.pos = func(self.maxes(), value)?;
            if bound.pos == self.maxes().len() {
                Ok(self.length().cast_signed())
            } else {
                bound.idx = func(&self.1[bound.pos], value)?;
                Ok(self.inner_mut().loc(&bound))
            }
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.0.clear();
        self.1.clear();
    }

    fn contains(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let key = self.2.bind(py).call1((&value,))?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(false),
            ops::Maxes::LenNEPos(mut bound) => {
                let first_sublist = &self.1[bound.pos];
                bound.idx = first_sublist.bisect_left(&key)?;
                let len_keys = self.1.len();
                let mut len_sublist = first_sublist.len();

                loop {
                    if self.1.iloc(&bound).bind(py).ne(&key)? {
                        return Ok(false);
                    }
                    if self.lists().iloc(&bound).bind(py).eq(value)? {
                        return Ok(true);
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(false);
                        }
                        len_sublist = self.1[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let key = self.2.bind(py).call1((value,))?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut bound) => {
                let first_sublist = &self.1[bound.pos];
                bound.idx = first_sublist.bisect_left(&key)?;
                let mut total = 0;
                let len_keys = self.1.len();
                let mut len_sublist = first_sublist.len();
                loop {
                    if self.1.iloc(&bound).bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if self.lists().iloc(&bound).bind(py).eq(value)? {
                        total += 1;
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = self.1[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, bounds: &mut Pos) -> PyResult<()> {
        self.1[bounds.pos].remove(bounds.idx);
        self.lists_mut()[bounds.pos].remove(bounds.idx);
        self.decrement_len();
        match ops::Delete::new(&self.1, self.load(), bounds) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.1[bounds.pos].last().unwrap().clone_ref(py);
                self.inner_mut().delete_on_idx(bounds, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if bounds.pos == 0 {
                    bounds.pos += 1;
                }
                let prev = bounds.pos - 1;
                let (left, right) = self.1.split_at_mut(bounds.pos);
                left[prev].append(&mut right[0]);

                let mut removed = self.0.lists[bounds.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.0.lists[prev].append(removed.as_mut());
                self.0.remove_pos(bounds);
                self.maxes_mut()[prev] = left[prev].last().unwrap().clone_ref(py);
                self.1.remove(bounds.pos);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes_mut()[bounds.pos] = self.1[bounds.pos].last().unwrap().clone_ref(py);
            }
            ops::Delete::Other => {
                self.inner_mut().remove_pos(bounds);
                self.1.remove(bounds.pos);
            }
        }
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let key = self.2.bind(py).call1((&value,))?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(()),
            ops::Maxes::LenNEPos(mut bound) => {
                let first_sublist = &self.1[bound.pos];
                bound.idx = first_sublist.bisect_left(&key)?;
                let len_keys = self.1.len();
                let mut len_sublist = first_sublist.len();
                loop {
                    if self.1.iloc(&bound).bind(py).ne(&key)? {
                        return Ok(());
                    }
                    if self.lists().iloc(&bound).bind(py).eq(&value)? {
                        return self.delete(py, &mut bound);
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return Ok(());
                        }
                        len_sublist = self.1[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new((self.1[pos]).len(), self.load(), self.idx()) {
            ops::Expand::PosLenGtLoad => {
                let half_keys = self.1[pos].split_off(self.0.load);
                let half = self.0.lists[pos].split_off(self.0.load);
                let new_max_at_pos = self.1[pos].last().unwrap().clone_ref(py);
                let last_max = half_keys.last().unwrap().clone_ref(py);
                self.inner_mut()
                    .expand_at_pos(pos, half, last_max, new_max_at_pos);
                self.1.insert(pos + 1, half_keys);
            }
            ops::Expand::IdxNotEmpty => self.inner_mut().expand_on_empty_idx(pos),
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
        let key = self.2.bind(py).call1((&value,))?;
        let (mut bound, start, mut stop) =
            ops::Index::new(self.inner(), &key, start, stop).into_res()?;
        stop -= 1;
        let v_left = &self.1[bound.pos];
        bound.idx = v_left.bisect_left(&key)?;
        let len_keys = self.1.len();
        let mut len_sublist = v_left.len();

        loop {
            if self.1.iloc(&bound).bind(py).ne(&key)? {
                return errors::not_in_list_err(value);
            }
            if self.lists().iloc(&bound).bind(py).eq(value)? {
                let loc = self.inner_mut().loc(&bound);
                if start <= loc && loc <= stop {
                    return Ok(loc);
                } else if loc > stop {
                    return errors::not_in_list_err(value);
                }
            }
            bound.idx += 1;
            if bound.idx == len_sublist {
                bound.pos += 1;
                if bound.pos == len_keys {
                    return errors::not_in_list_err(value);
                }
                len_sublist = self.1[bound.pos].len();
                bound.idx = 0;
            }
        }
    }

    fn finalize_update(&mut self, py: Python<'_>, values: &[Py<PyAny>]) -> PyResult<()> {
        self.inner_mut().extend_lists(py, values);
        let key_fn = self.2.bind(py);
        self.lists()
            .iter()
            .map(|list| {
                list.iter()
                    .map(|x| key_fn.call1((x,)).map(Bound::unbind))
                    .collect()
            })
            .collect::<PyResult<Vec<_>>>()
            .map(|mut vec| self.1.append(vec.as_mut()))?;
        self.1
            .iter()
            .map(|x| x.last().unwrap().clone_ref(py))
            .pipe(|it| self.0.maxes.extend(it));
        self.set_len(values.len());
        self.idx_mut().clear();
        Ok(())
    }
    fn remove(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let key = self.2.bind(py).call1((&value,))?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => errors::not_in_list_err(value),
            ops::Maxes::LenNEPos(mut bound) => {
                let first_sublist = &self.1[bound.pos];
                bound.idx = first_sublist.bisect_left(&key)?;
                let len_keys = self.1.len();
                let mut len_sublist = first_sublist.len();

                loop {
                    if self.1.iloc(&bound).bind(py).ne(&key)? {
                        return errors::not_in_list_err(value);
                    }
                    if self.lists().iloc(&bound).bind(py).eq(value)? {
                        return self.delete(py, &mut bound);
                    }
                    bound.idx += 1;
                    if bound.idx == len_sublist {
                        bound.pos += 1;
                        if bound.pos == len_keys {
                            return errors::not_in_list_err(value);
                        }
                        len_sublist = self.1[bound.pos].len();
                        bound.idx = 0;
                    }
                }
            }
        }
    }

    fn update(&mut self, py: Python<'_>, values: VecPy) -> PyResult<()> {
        let key_fn = &self.2.clone_ref(py).into_bound(py);
        update_list_by(self, py, values, |a, b| py_cmp_by_key(a, b, key_fn))
    }
}
