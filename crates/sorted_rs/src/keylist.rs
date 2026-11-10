use crate::{
    KeyList,
    bounds::{Loc, RangeIdxs, RawIrangeSpecs},
    errors,
    inner::Values,
    iter::{IrangeSpecs, builder::RawRangeIdxs},
    list_update::{ListAdd, Updater},
    ops,
    traits::ListsMethods,
    types::VecPy,
};
use itertools::Itertools;
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
impl KeyList {
    fn extract_key<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.2.bind(value.py()).call1((&value,))
    }
}
impl ListsMethods for KeyList {
    fn keys(&self) -> &Values {
        &self.1
    }
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::new(self.2.clone_ref(py));
        new_inst.extend(py, values.into())?;
        Ok(new_inst)
    }
    fn irange<'py>(
        &self,
        py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeSpecs<'py>> {
        IrangeSpecs::with_keys(py, specs, &self.2)
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.extract_key(&value)?;
        let py = value.py();
        match self.try_bisect_right(&key)? {
            ops::Bisect::Empty => {
                self.values.push(vec![value.unbind()]);
                let v = key.unbind();
                self.1.push(vec![v.clone_ref(py)]);
                self.maxes.push(v);
            }
            ops::Bisect::LenEQPos => {
                let pos = self.maxes.len() - 1;
                let v = key.unbind();
                self.values[pos].push(value.unbind());
                self.1[pos].push(v.clone_ref(py));
                self.maxes[pos] = v;
                self.expand(py, pos);
            }
            ops::Bisect::LenNEPos(loc) => {
                self.values.loc_insert(&loc, value.unbind());
                self.1.loc_insert(&loc, key.unbind());
                self.expand(py, loc.pos);
            }
        }
        self.len += 1;
        Ok(())
    }

    #[inline]
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.extract_key(value)
            .and_then(|x| self.bisect_by_key::<LEFT>(&x))
    }
    #[inline]
    fn clear(&mut self, _py: Python<'_>) {
        self.0.clear();
        self.1.clear();
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let key = self.2.bind(py).call1((value,))?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(0),
            ops::Bisect::LenNEPos(mut loc) => {
                let mut total = 0;
                let len_keys = self.keys().len();
                let mut len_sublist = self.keys()[loc.pos].len();
                loop {
                    if self.keys().loc(&loc).bind(py).ne(&key)? {
                        return Ok(total);
                    }
                    if self.values.loc(&loc).bind(py).eq(value)? {
                        total += 1;
                    }
                    loc.idx += 1;
                    if loc.idx == len_sublist {
                        loc.pos += 1;
                        if loc.pos == len_keys {
                            return Ok(total);
                        }
                        len_sublist = self.keys().loc_len(&loc);
                        loc.idx = 0;
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, mut loc: Loc) -> PyResult<()> {
        self.1.loc_remove(&loc);
        self.values.loc_remove(&loc);
        self.len -= 1;
        match ops::SubListState::new(self, loc.pos) {
            ops::SubListState::LenSupToHalfLoad => {
                self.delete_on_len_sup_to_half_load(py, loc);
            }
            ops::SubListState::ManySublists => {
                if loc.pos == 0 {
                    loc.pos += 1;
                }
                let prev = loc.pos - 1;
                let (left, right) = self.1.split_at_mut(loc.pos);
                left[prev].append(&mut right[0]);
                let mut removed = self.0.values[loc.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.0.values[prev].append(removed.as_mut());
                self.0.remove_pos(&loc);
                self.maxes[prev] = left[prev].last().unwrap().clone_ref(py);
                self.1.remove(loc.pos);
                self.expand(py, prev);
            }
            ops::SubListState::NonEmpty => {
                self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
            }
            ops::SubListState::Empty => {
                self.remove_pos(&loc);
                self.1.remove(loc.pos);
            }
        }
        Ok(())
    }
    fn expand_splitted(&mut self, py: Python<'_>, pos: usize) -> (VecPy, Py<PyAny>) {
        let half_keys = self.1[pos].split_off(*self.0.load);
        let half = self.0.values[pos].split_off(*self.0.load);
        let last_max = half_keys.last().unwrap().clone_ref(py);
        self.1.insert(pos + 1, half_keys);
        (half, last_max)
    }

    fn index(&mut self, value: &Bound<'_, PyAny>, indices: RawRangeIdxs) -> PyResult<usize> {
        let py = value.py();
        let indices = RangeIdxs::from(indices);
        match indices.into_index(self) {
            None => Err(errors::not_in_list(value)),
            Some(indices) => {
                let key = self.extract_key(value)?;
                let mut loc = self.bisect_or_err(&key)?;
                let len_keys = self.keys().len();
                loop {
                    if self.keys().loc(&loc).bind(py).ne(&key)? {
                        return Err(errors::not_in_list(value));
                    }
                    if self.values.loc(&loc).bind(py).eq(value)? {
                        let loc = self.loc(&loc);
                        if indices.start <= loc && loc < indices.stop {
                            return Ok(loc);
                        } else if loc >= indices.stop {
                            return Err(errors::not_in_list(value));
                        }
                    }
                    loc.idx += 1;
                    if loc.idx == self.keys().loc_len(&loc) {
                        loc.pos += 1;
                        if loc.pos == len_keys {
                            return Err(errors::not_in_list(value));
                        }
                        loc.idx = 0;
                    }
                }
            }
        }
    }

    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        let key = self.extract_key(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(None),
            ops::Bisect::LenNEPos(mut loc) => {
                let py = value.py();
                let len_keys = self.keys().len();
                loop {
                    if self.keys().loc(&loc).bind(py).ne(&key)? {
                        return Ok(None);
                    }
                    if self.values.loc(&loc).bind(py).eq(value)? {
                        return Ok(Some(loc));
                    }
                    loc.idx += 1;
                    if loc.idx == self.keys().loc_len(&loc) {
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
    fn finalize_update(&mut self, py: Python<'_>, it: impl Iterator<Item = VecPy>) -> PyResult<()> {
        let key_fn = self.2.clone_ref(py).into_bound(py);
        it.map(|val_chunk| {
            val_chunk
                .iter()
                .map(|v| key_fn.call1((v,)))
                .map_ok(Bound::unbind)
                .collect::<PyResult<VecPy>>()
                .map(|key_chunk| (val_chunk, key_chunk))
        })
        .try_for_each(|res| {
            res.map(|(val_chunk, key_chunk)| {
                self.maxes.push(key_chunk.last().unwrap().clone_ref(py));
                self.values.push(val_chunk);
                self.1.push(key_chunk);
            })
        })
    }
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()> {
        let key_fn = &self.2.clone_ref(py).into_bound(py);
        Updater::new(py, self, Some(key_fn)).merge_with(values, |a, b| {
            key_fn
                .call1((a,))
                .unwrap()
                .compare_fast(&key_fn.call1((b,)).unwrap())
                .unwrap()
        })
    }
}
