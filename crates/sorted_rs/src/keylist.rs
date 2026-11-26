use crate::{
    KeyList,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    chunks::ChunksWithKeys,
    iter::IrangeBounds,
    ops,
    search::Search,
    sort::WithKey,
    traits::ListsMethods,
};
use pyo3::prelude::*;
use tap::prelude::*;
impl ListsMethods for KeyList {
    type Chunks = ChunksWithKeys;
    type Key<'py> = WithKey<'py>;
    fn key_func<'py>(&self, py: Python<'py>) -> Self::Key<'py> {
        self.func.clone_ref(py).into_bound(py).pipe(WithKey)
    }
    fn new_empty(&self, py: Python<'_>) -> Self {
        Self::new(self.func.clone_ref(py))
    }
    fn copy(&self, py: Python<'_>) -> Self {
        Self {
            inner: self.inner.clone_ref(py),
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
        self.append_item(value.py(), (key.unbind(), value.unbind()))
    }

    #[inline]
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.call_fn(value)
            .and_then(|x| self.bisect_by_key::<LEFT>(&x))
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let key = self.call_fn(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen(_) => Ok(0),
            ops::Bisect::AtLoc(loc) => Search::new(self, loc, value, &key).count(),
        }
    }
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        let key = self.call_fn(value)?;
        Search::on_index(self, value, &key)?.index_by_key(indices)
    }
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        let key = self.call_fn(value)?;
        match self.try_bisect_left(&key)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen(_) => Ok(None),
            ops::Bisect::AtLoc(loc) => Search::new(self, loc, value, &key).find(),
        }
    }
}
