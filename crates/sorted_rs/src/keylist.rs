use crate::{
    Bounds,
    bisect::Bisect,
    bounds::Loc,
    cmp::py_cmp_by_key,
    errors,
    getters::{InnerGetter, ListDataGetters},
    impl_inner_getter,
    inner::{InnerData, VecPy},
    ops,
    traits::{ListsDataMethods, NestedVec, update_list_by},
};
use pyo3::{prelude::*, types::PyString};
use tap::prelude::*;
pub struct KeysListsData(InnerData, pub Vec<VecPy>, pub Py<PyAny>);

impl KeysListsData {
    #[must_use]
    pub fn new(key: Py<PyAny>) -> Self {
        Self(InnerData::default(), Vec::default(), key)
    }
    fn extract_key<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.2.bind(value.py()).call1((&value,))
    }
}
impl_inner_getter!(KeysListsData);
impl ListsDataMethods for KeysListsData {
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::new(self.2.clone_ref(py));
        new_inst.extend(py, values)?;
        Ok(new_inst)
    }
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
        let key = self.extract_key(value.bind(py))?;
        match ops::Maxes::right(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                self.lists_mut().push(vec![value]);
                let v = key.unbind();
                self.1.push(vec![v.clone_ref(py)]);
                self.maxes_mut().push(v);
            }
            ops::Maxes::LenEQPos(mut loc) => {
                loc.pos -= 1;
                let v = key.unbind();
                self.lists_mut().loc_push(&loc, value);
                self.1.loc_push(&loc, v.clone_ref(py));
                self.maxes_mut()[loc.pos] = v;
                self.expand(py, loc.pos);
            }
            ops::Maxes::LenNEPos(mut loc) => {
                let v = &self.1[loc.pos];
                loc.idx = v.bisect_right(&key)?;
                self.lists_mut().loc_insert(&loc, value);
                self.1.loc_insert(&loc, key.unbind());
                self.expand(py, loc.pos);
            }
        }
        self.increment_len();
        Ok(())
    }

    fn bisect(
        &mut self,
        value: &Bound<'_, PyAny>,
        func: fn(&[pyo3::Py<pyo3::PyAny>], &Bound<'_, PyAny>) -> PyResult<usize>,
    ) -> PyResult<usize> {
        if self.maxes().is_empty() {
            Ok(0)
        } else {
            let mut loc = Loc::new(0, 0);
            loc.pos = func(self.maxes(), value)?;
            if loc.pos == self.maxes().len() {
                Ok(self.length())
            } else {
                loc.idx = func(&self.1[loc.pos], value)?;
                Ok(self.inner_mut().loc(&loc))
            }
        }
    }
    #[inline]
    fn clear(&mut self) {
        self.0.clear();
        self.1.clear();
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let key = self.2.bind(py).call1((value,))?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut loc) => {
                let first_sublist = &self.1[loc.pos];
                loc.idx = first_sublist.bisect_left(&key)?;
                let mut total = 0;
                let len_keys = self.1.len();
                let mut len_sublist = first_sublist.len();
                loop {
                    if self.1.loc(&loc).bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if self.lists().loc(&loc).bind(py).eq(value)? {
                        total += 1;
                    }
                    loc.idx += 1;
                    if loc.idx == len_sublist {
                        loc.pos += 1;
                        if loc.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = self.1.loc_len(&loc);
                        loc.idx = 0;
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, loc: &mut Loc) -> PyResult<()> {
        self.1.loc_remove(loc);
        self.lists_mut().loc_remove(loc);
        self.decrement_len();
        match ops::Delete::new(&self.1, self.load(), loc) {
            ops::Delete::PosSupToLoad => {
                let max_at_pos = self.1.loc_last(loc).clone_ref(py);
                self.inner_mut().delete_on_idx(loc, max_at_pos);
            }
            ops::Delete::DataLenGTOne => {
                if loc.pos == 0 {
                    loc.pos += 1;
                }
                let prev = loc.pos - 1;
                let (left, right) = self.1.split_at_mut(loc.pos);
                left[prev].append(&mut right[0]);

                let mut removed = self.0.lists[loc.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.0.lists[prev].append(removed.as_mut());
                self.0.remove_pos(loc);
                self.maxes_mut()[prev] = left[prev].last().unwrap().clone_ref(py);
                self.1.remove(loc.pos);
                self.expand(py, prev);
            }
            ops::Delete::LenPosNotZero => {
                self.maxes_mut()[loc.pos] = self.1.loc_last(loc).clone_ref(py);
            }
            ops::Delete::Other => {
                self.inner_mut().remove_pos(loc);
                self.1.remove(loc.pos);
            }
        }
        Ok(())
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
    ) -> PyResult<usize> {
        let py = value.py();
        let key = self.extract_key(value)?;
        let (mut loc, start, mut stop) =
            ops::Index::new(self.inner(), &key, start, stop).into_res()?;
        stop -= 1;
        let v_left = &self.1[loc.pos];
        loc.idx = v_left.bisect_left(&key)?;
        let len_keys = self.1.len();
        let mut len_sublist = v_left.len();

        loop {
            if self.1.loc(&loc).bind(py).ne(&key)? {
                return Err(errors::not_in_list(&value.repr()?));
            }
            if self.lists().loc(&loc).bind(py).eq(value)? {
                let loc = self.inner_mut().loc(&loc);
                if start <= loc && loc <= stop {
                    return Ok(loc);
                } else if loc > stop {
                    return Err(errors::not_in_list(&value.repr()?));
                }
            }
            loc.idx += 1;
            if loc.idx == len_sublist {
                loc.pos += 1;
                if loc.pos == len_keys {
                    return Err(errors::not_in_list(&value.repr()?));
                }
                len_sublist = self.1.loc_len(&loc);
                loc.idx = 0;
            }
        }
    }

    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        let key = self.extract_key(value)?;
        match ops::Maxes::left(self.maxes(), &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(None),
            ops::Maxes::LenNEPos(mut loc) => {
                loc.idx = self.1[loc.pos].bisect_left(&key)?;
                let py = value.py();
                let len_keys = self.1.len();
                loop {
                    if self.1.loc(&loc).bind(py).ne(&key)? {
                        return Ok(None);
                    }
                    if self.lists().loc(&loc).bind(py).eq(value)? {
                        return Ok(Some(loc));
                    }
                    loc.idx += 1;
                    if loc.idx == self.1.loc_len(&loc) {
                        loc.pos += 1;

                        if loc.pos == len_keys {
                            return Ok(None);
                        }
                        loc.idx = 0;
                    }
                }
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
    fn repr(&self, py: Python<'_>, name: Bound<'_, PyString>) -> PyResult<String> {
        let key_repr = self.2.bind(py).repr()?;
        self.inner()
            .as_pylist(py)?
            .repr()
            .map(|repr| format!("{name}({repr}, key={key_repr})"))
    }
    fn extend(&mut self, py: Python<'_>, values: VecPy) -> PyResult<()> {
        let key_fn = &self.2.clone_ref(py).into_bound(py);
        update_list_by(self, py, values, |a, b| py_cmp_by_key(a, b, key_fn))
    }
}
