use crate::{
    KeyList,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    delete,
    item::ItemRef,
    iter::IrangeBounds,
    ops,
    search::Search,
    sort::{Chunk, ChunkKey, WithKey},
    sublist::{Chunks, SubList},
    traits::ListsMethods,
};
use pyo3::prelude::*;
use tap::prelude::*;
impl ListsMethods for KeyList {
    type Chunk = ChunkKey;
    type Item = (Py<PyAny>, Py<PyAny>);
    type Key<'py> = WithKey<'py>;
    fn key_func<'py>(&self, py: Python<'py>) -> Self::Key<'py> {
        self.func.clone_ref(py).into_bound(py).pipe(WithKey)
    }
    fn keys(&self) -> &Chunks {
        &self.keys
    }
    fn keys_mut(&mut self) -> &mut Chunks {
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
    fn insert_at(&mut self, loc: &Loc, (key, value): Self::Item) {
        self.keys_mut().loc_insert(loc, key);
        self.values.loc_insert(loc, value);
    }
    fn loc_item(&self, loc: &Loc) -> impl ItemRef<Owned = Self::Item> {
        (self.keys.loc(loc), self.values.loc(loc))
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let key = self.call_fn(&value)?;
        let py = value.py();
        match self.try_bisect_right(&key)? {
            ops::Bisect::Empty => {
                self.values.push(value.unbind().pipe(SubList::new));
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
    fn delete_values(&mut self, loc: Loc) {
        self.values.remove(loc.pos);
    }
    fn delete_merge_keys(&mut self, loc: Loc, merge: delete::Merge) {
        merge.remove(&mut self.keys, loc);
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
    fn loc_remove(&mut self, loc: Loc) {
        self.keys.loc_remove(&loc);
        self.values.loc_remove(&loc);
    }
    fn split_values(&mut self, pos: usize, load: usize) {
        let half = self.values[pos].split_off(load);
        self.values.insert(pos + 1, half);
    }
}
