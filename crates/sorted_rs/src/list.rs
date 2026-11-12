use pyo3::prelude::*;
use pyo3_ext::prelude::*;
use tap::{Conv, Pipe};

use crate::{
    List,
    bounds::{Bounds, Loc, RawIrangeSpecs},
    inner::Values,
    iter::IrangeBounds,
    list_update::{ListAdd, Updater},
    ops,
    search::Search,
    traits::ListsMethods,
    types::VecPy,
};

impl ListsMethods for List {
    fn keys(&self) -> &Values {
        &self.values
    }
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::default();
        new_inst.extend(py, values.into())?;
        Ok(new_inst)
    }
    fn irange<'py>(
        &self,
        _py: Python<'py>,
        specs: RawIrangeSpecs<'py>,
    ) -> PyResult<IrangeBounds<'py>> {
        specs.conv::<IrangeBounds>().pipe(Ok)
    }
    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        match self.try_bisect_right(&value)? {
            ops::Bisect::Empty => {
                let unbounded = value.unbind();
                self.values.push(vec![unbounded.clone_ref(py)]);
                self.maxes.push(unbounded);
            }
            ops::Bisect::LenEQPos => {
                let pos = self.maxes.len() - 1;
                let unbounded = value.unbind();
                self.values[pos].push(unbounded.clone_ref(py));
                self.maxes[pos] = unbounded;
                self.expand(py, pos);
            }
            ops::Bisect::LenNEPos(loc) => {
                self.values[loc.pos].insert(loc.idx, value.unbind());
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
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(0),
            ops::Bisect::LenNEPos(left) => match self.try_bisect_right(value)? {
                ops::Bisect::Empty => unreachable!(),
                ops::Bisect::LenEQPos => Ok(self.len() - left.into_index(self)),
                ops::Bisect::LenNEPos(right) => {
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
                if loc.pos == 0 {
                    loc.pos += 1;
                }
                let prev = loc.pos - 1;
                let mut removed = self.values[loc.pos]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>();
                self.values[prev].append(removed.as_mut());
                self.remove_pos(&loc);
                self.maxes[prev] = self.keys()[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::SubListState::NonEmpty => {
                self.maxes[loc.pos] = self.keys().loc_last(&loc).clone_ref(py);
            }
            ops::SubListState::Empty => self.remove_pos(&loc),
        }
        Ok(())
    }
    fn expand_splitted(&mut self, py: Python<'_>, pos: usize) -> (VecPy, Py<PyAny>) {
        let half = self.0.values[pos].split_off(*self.0.load);
        let last_max = half.last().unwrap().clone_ref(py);
        (half, last_max)
    }
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()> {
        Updater::new(py, self, None)
            .merge_with(values, |a, b| a.bind(py).compare_fast(b.bind(py)).unwrap())
    }
    fn index_inner(&mut self, value: &Bound<'_, PyAny>, indices: Bounds<usize>) -> PyResult<usize> {
        Search::on_index(self, value, value)?.index_by_val(indices)
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        match self.try_bisect_left(value)? {
            ops::Bisect::Empty | ops::Bisect::LenEQPos => Ok(None),
            ops::Bisect::LenNEPos(loc) => match self.values.loc(&loc).bind(value.py()).eq(value) {
                Ok(true) => Ok(Some(loc)),
                Ok(false) => Ok(None),
                Err(err) => Err(err),
            },
        }
    }
    fn finalize_update(&mut self, py: Python<'_>, it: impl Iterator<Item = VecPy>) -> PyResult<()> {
        it.for_each(|chunk| {
            self.maxes.push(chunk.last().unwrap().clone_ref(py));
            self.values.push(chunk);
        });
        Ok(())
    }
}
