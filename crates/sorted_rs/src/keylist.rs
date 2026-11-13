use crate::{
    KeyList,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    inner::Values,
    iter::IrangeBounds,
    list_update::{ListAdd, Updater},
    ops,
    search::Search,
    traits::ListsMethods,
    types::VecPy,
};
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
impl ListsMethods for KeyList {
    fn keys(&self) -> &Values {
        &self.keys
    }
    fn keys_mut(&mut self) -> &mut Values {
        &mut self.keys
    }
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::new(self.func.clone_ref(py));
        new_inst.extend(py, values.into())?;
        Ok(new_inst)
    }
    fn irange<'py>(
        &self,
        py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeBounds<'py>> {
        IrangeBounds::with_keys(specs, self.func.bind(py))
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.call_fn(&value)?;
        let py = value.py();
        match self.try_bisect_right(&key)? {
            ops::Bisect::Empty => {
                self.values.push(vec![value.unbind()]);
                self.push_chunk(py, vec![key.unbind()]);
            }
            ops::Bisect::LenEQPos => {
                let pos = self.maxes.len() - 1;
                self.values[pos].push(value.unbind());
                self.push_val(py, pos, key.unbind());
                self.expand(py, pos);
            }
            ops::Bisect::LenNEPos(loc) => {
                self.values.loc_insert(&loc, value.unbind());
                self.keys_mut().loc_insert(&loc, key.unbind());
                self.expand(py, loc.pos);
            }
        }
        self.len += 1;
        Ok(())
    }

    #[inline]
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.call_fn(value)
            .and_then(|x| self.bisect_by_key::<LEFT>(&x))
    }
    #[inline]
    fn clear(&mut self, _py: Python<'_>) {
        self.inner.clear();
        self.keys.clear();
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let key = self.call_fn(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(0),
            ops::Bisect::LenNEPos(loc) => Search::new(self, loc, value, &key).count(),
        }
    }

    fn delete(&mut self, py: Python<'_>, mut loc: Loc) -> PyResult<()> {
        self.keys.loc_remove(&loc);
        self.values.loc_remove(&loc);
        self.len -= 1;
        match ops::SubListState::new(self, loc.pos) {
            ops::SubListState::LenSupToHalfLoad => {
                self.delete_on_len_sup_to_half_load(py, loc);
            }
            ops::SubListState::ManySublists => {
                let prev = self.delete_many(&mut loc);
                let (left, right) = self.keys.split_at_mut(loc.pos);
                left[prev].append(&mut right[0]);
                self.maxes[prev] = left[prev].last().unwrap().clone_ref(py);
                self.keys.remove(loc.pos);
                self.expand(py, prev);
            }
            ops::SubListState::NonEmpty => {
                self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
            }
            ops::SubListState::Empty => {
                self.remove_at(&loc);
                self.keys.remove(loc.pos);
            }
        }
        Ok(())
    }
    fn expand_splitted(&mut self, py: Python<'_>, pos: usize) -> (VecPy, Py<PyAny>) {
        let load = *self.load;
        let half_keys = self.keys[pos].split_off(load);
        let half = self.values[pos].split_off(load);
        let last_max = half_keys.last().unwrap().clone_ref(py);
        self.keys.insert(pos + 1, half_keys);
        (half, last_max)
    }

    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        let key = self.call_fn(value)?;
        Search::on_index(self, value, &key)?.index_by_key(indices)
    }

    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        let key = self.call_fn(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(None),
            ops::Bisect::LenNEPos(loc) => Search::new(self, loc, value, &key).find(),
        }
    }
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()> {
        let key_fn = &self.func.clone_ref(py).into_bound(py);
        Updater::new(py, self, Some(key_fn)).merge_with(values, |a, b| {
            key_fn
                .call1((a,))
                .unwrap()
                .compare_fast(&key_fn.call1((b,)).unwrap())
                .unwrap()
        })
    }
}
