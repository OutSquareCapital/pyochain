use crate::{
    KeyList,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    inner::Values,
    iter::IrangeBounds,
    ops,
    search::Search,
    sort::{Chunk, WithKey},
    traits::ListsMethods,
    types::VecPy,
};
use pyo3::prelude::*;
use tap::prelude::*;
impl ListsMethods for KeyList {
    type Chunk = (VecPy, VecPy);
    type Item = (Py<PyAny>, Py<PyAny>);
    type Key<'py> = WithKey<'py>;
    fn key_func<'py>(&self, py: Python<'py>) -> Self::Key<'py> {
        self.func.clone_ref(py).into_bound(py).pipe(WithKey)
    }
    fn keys(&self) -> &Values {
        &self.keys
    }
    fn keys_mut(&mut self) -> &mut Values {
        &mut self.keys
    }
    fn new_empty(&self, py: Python<'_>) -> Self {
        Self::new(self.func.clone_ref(py))
    }
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            inner: self.inner.clone_ref(py),
            keys: self.keys.clone_ref(py),
            func: self.func.clone_ref(py),
        }
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
                vec![key.unbind()].push_into(py, self);
            }
            ops::Bisect::MaxesLen => {
                let pos = self.maxes.len() - 1;
                self.values[pos].push(value.unbind());
                self.push_val(py, key.unbind());
            }
            ops::Bisect::AtLoc(loc) => {
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
            ops::Bisect::Empty | ops::Bisect::MaxesLen => Ok(0),
            ops::Bisect::AtLoc(loc) => Search::new(self, loc, value, &key).count(),
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
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        let key = self.call_fn(value)?;
        Search::on_index(self, value, &key)?.index_by_key(indices)
    }

    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        let key = self.call_fn(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen => Ok(None),
            ops::Bisect::AtLoc(loc) => Search::new(self, loc, value, &key).find(),
        }
    }
    fn split_values(&mut self, pos: usize, load: usize) {
        let half = self.values[pos].split_off(load);
        self.values.insert(pos + 1, half);
    }
}
