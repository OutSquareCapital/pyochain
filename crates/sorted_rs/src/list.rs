use pyo3::prelude::*;
use tap::{Conv, Pipe};

use crate::{
    List,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    inner::Values,
    iter::IrangeBounds,
    ops,
    search::Search,
    sort::{Chunk, Identity},
    traits::ListsMethods,
    types::VecPy,
};

impl ListsMethods for List {
    type Chunk = VecPy;
    type Item = Py<PyAny>;
    type Key<'py> = Identity<'py>;
    fn keys(&self) -> &Values {
        &self.values
    }
    fn new_empty(&self, _py: Python<'_>) -> Self {
        Self::default()
    }
    fn copy(&self, py: Python<'_>) -> Self {
        Self(self.0.clone_ref(py))
    }
    fn keys_mut(&mut self) -> &mut Values {
        &mut self.values
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
    fn insert_at(&mut self, loc: &Loc, item: Self::Item) {
        self.keys_mut().loc_insert(loc, item);
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        match self.try_bisect_right(&value)? {
            ops::Bisect::Empty => {
                vec![value.unbind()].push_into(py, self);
            }
            ops::Bisect::MaxesLen => {
                self.push_val(py, value.unbind());
            }
            ops::Bisect::AtLoc(loc) => {
                self.keys_mut().loc_insert(&loc, value.unbind());
                self.expand(py, loc.pos);
            }
        }
        self.len += 1;
        Ok(())
    }
    #[inline]
    fn bisect<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        self.bisect_by_key::<LEFT>(value)
    }
    #[inline]
    fn clear(&mut self, _py: Python<'_>) {
        self.0.clear();
    }
    fn count(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.try_bisect_left(value)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen => Ok(0),
            ops::Bisect::AtLoc(left) => match self.try_bisect_right(value)? {
                ops::Bisect::Empty => unreachable!(),
                ops::Bisect::MaxesLen => Ok(self.len() - left.into_index(self)),
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

    fn delete(&mut self, py: Python<'_>, mut loc: Loc) -> PyResult<()> {
        self.values.loc_remove(&loc);
        self.len -= 1;
        match ops::SubListState::new(self, loc.pos) {
            ops::SubListState::LenSupToHalfLoad => {
                self.delete_on_len_sup_to_half_load(py, loc);
            }
            ops::SubListState::ManySublists => {
                let prev = self.delete_many(&mut loc);
                self.maxes[prev] = self.values[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::SubListState::NonEmpty => {
                self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
            }
            ops::SubListState::Empty => {
                self.remove_at(&loc);
            }
        }
        Ok(())
    }
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        Search::on_index(self, value, value)?.index_by_val(indices)
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        match self.try_bisect_left(value)? {
            ops::Bisect::Empty | ops::Bisect::MaxesLen => Ok(None),
            ops::Bisect::AtLoc(loc) => match self.values.loc(&loc).bind(value.py()).eq(value) {
                Ok(true) => Ok(Some(loc)),
                Ok(false) => Ok(None),
                Err(err) => Err(err),
            },
        }
    }
    fn split_values(&mut self, _pos: usize, _load: usize) {}
}
