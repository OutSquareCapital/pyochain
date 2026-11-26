use pyo3::prelude::*;
use tap::{Conv, Pipe};

use crate::{
    List,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    chunks::Chunks,
    iter::IrangeBounds,
    ops,
    search::Search,
    sort::Identity,
    traits::ListsMethods,
};

impl ListsMethods for List {
    type Chunks = Chunks;
    type Key<'py> = Identity<'py>;
    fn new_empty(&self, _py: Python<'_>) -> Self {
        Self::default()
    }
    fn copy(&self, py: Python<'_>) -> Self {
        Self(self.0.clone_ref(py))
    }
    fn key_func<'py>(&self, py: Python<'py>) -> Self::Key<'py> {
        Identity(py)
    }
    fn irange<'py>(
        &self,
        _py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeBounds<'py>> {
        specs.conv::<IrangeBounds>().pipe(Ok)
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.append_item(value.py(), value.unbind())
    }
    #[inline]
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.bisect_by_key::<LEFT>(value)
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.try_bisect_left(value)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen(_) => Ok(0),
            ops::Bisect::AtLoc(left) => match self.try_bisect_right(value)? {
                ops::Bisect::Empty => unreachable!(),
                ops::Bisect::MaxesLen(_) => Ok(self.len() - left.into_index(self)),
                ops::Bisect::AtLoc(right) => {
                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_idx = right.into_index(self);
                        let left_idx = left.into_index(self);
                        Ok(right_idx - left_idx)
                    }
                }
            },
        }
    }
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        Search::on_index(self, value, value)?.index_by_val(indices)
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        match self.try_bisect_left(value)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen(_) => Ok(None),
            ops::Bisect::AtLoc(loc) => match self.chunks.get_at(loc).bind(value.py()).eq(value) {
                Ok(true) => Ok(Some(loc)),
                Ok(false) => Ok(None),
                Err(err) => Err(err),
            },
        }
    }
}
