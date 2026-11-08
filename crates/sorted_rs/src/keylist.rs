use crate::{
    Bounds, KeyList,
    bisect::Bisect,
    bounds::{IrangeSpecs, Loc, RangeIdxs, RawIrangeSpecs, RawRangeIdxs},
    errors,
    list_update::ListAdd,
    ops,
    traits::{ListsMethods, NestedVec},
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
    fn keys(&self) -> &[VecPy] {
        &self.1
    }
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::new(self.2.clone_ref(py));
        new_inst.extend(py, values.into())?;
        Ok(new_inst)
    }
    fn irange(&self, specs: RawIrangeSpecs<'_>) -> Option<PyResult<Bounds>> {
        IrangeSpecs::with_keys(specs, &self.2)
            .map(|x| x.into_bounds(self))
            .transpose()
            .map(Result::flatten)
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.extract_key(&value)?;
        let py = value.py();
        match ops::Maxes::right(&self.maxes, &key) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                self.values.push(vec![value.unbind()]);
                let v = key.unbind();
                self.1.push(vec![v.clone_ref(py)]);
                self.maxes.push(v);
            }
            ops::Maxes::LenEQPos(mut loc) => {
                loc.pos -= 1;
                let v = key.unbind();
                self.values.loc_push(&loc, value.unbind());
                self.1.loc_push(&loc, v.clone_ref(py));
                self.maxes[loc.pos] = v;
                self.expand(py, loc.pos);
            }
            ops::Maxes::LenNEPos(mut loc) => {
                let v = &self.keys()[loc.pos];
                loc.idx = v.bisect_right(&key)?;
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
        match ops::Maxes::left(&self.maxes, &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut loc) => {
                let first_sublist = &self.keys()[loc.pos];
                loc.idx = first_sublist.bisect_left(&key)?;
                let mut total = 0;
                let len_keys = self.keys().len();
                let mut len_sublist = first_sublist.len();
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
        match ops::Delete::new(self, &loc) {
            ops::Delete::LenAtPosSupToHalfLoad => {
                self.delete_on_len_sup_to_half_load(py, loc);
            }
            ops::Delete::ListsLenGTOne => {
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
            ops::Delete::LenAtPosNotZero => {
                self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
            }
            ops::Delete::Other => {
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
                let mut loc = Loc::try_from_index(self, &key)?;
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
        match ops::Maxes::left(&self.maxes, &key) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(None),
            ops::Maxes::LenNEPos(mut loc) => {
                loc.idx = self.keys()[loc.pos].bisect_left(&key)?;
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
        values.merge_with(
            self,
            py,
            |a, b| {
                key_fn
                    .call1((a,))
                    .unwrap()
                    .compare_fast(&key_fn.call1((b,)).unwrap())
                    .unwrap()
            },
            Some(key_fn),
        )
    }
}
