use pyo3::prelude::*;
use pyo3_ext::prelude::*;

use crate::{
    ListsData, RangeIdxs,
    bisect::Bisect,
    bounds::{Bounds, IrangeSpecs, Loc},
    errors,
    list_update::ListAdd,
    ops,
    traits::{ListsDataMethods, NestedVec},
    types::VecPy,
};

impl ListsDataMethods for ListsData {
    fn keys(&self) -> &[VecPy] {
        &self.values
    }
    fn as_owned_from(&self, py: Python<'_>, values: VecPy) -> PyResult<Self> {
        let mut new_inst = Self::default();
        new_inst.extend(py, values.into())?;
        Ok(new_inst)
    }
    fn irange_specs(&self, specs: IrangeSpecs<'_>) -> PyResult<Bounds> {
        specs.into_bounds(self)
    }

    fn append(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        match ops::Maxes::right(&self.0.maxes, &value) {
            ops::Maxes::BisectErr(err) => return Err(err),
            ops::Maxes::Empty => {
                let unbounded = value.unbind();
                self.0.values.push(vec![unbounded.clone_ref(py)]);
                self.0.maxes.push(unbounded);
            }
            ops::Maxes::LenEQPos(mut loc) => {
                loc.pos -= 1;
                let unbounded = value.unbind();
                self.0.values.loc_push(&loc, unbounded.clone_ref(py));
                self.0.maxes[loc.pos] = unbounded;
                self.expand(py, loc.pos);
            }
            ops::Maxes::LenNEPos(loc) => {
                let res = self.0.values[loc.pos].bisect_right(&value)?;
                self.0.values[loc.pos].insert(res, value.unbind());
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
        match ops::Maxes::left(&self.maxes, value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(0),
            ops::Maxes::LenNEPos(mut left) => {
                let mut right = Loc::default();
                left.idx = self.values[left.pos].bisect_left(value)?;
                right.pos = self.maxes.bisect_right(value)?;

                if right.pos == self.maxes.len() {
                    let left_loc = self.loc(&left);
                    Ok(self.len - left_loc)
                } else {
                    right.idx = self.values[right.pos].bisect_right(value)?;

                    if left.pos == right.pos {
                        Ok(right.idx - left.idx)
                    } else {
                        let right_loc = self.loc(&right);
                        let left_loc = self.loc(&left);
                        Ok(right_loc - left_loc)
                    }
                }
            }
        }
    }

    fn delete(&mut self, py: Python<'_>, mut loc: Loc) -> PyResult<()> {
        self.values.loc_remove(&loc);
        self.len -= 1;
        match ops::Delete::new(&self.values, self.load, &loc) {
            ops::Delete::LenAtPosSupToHalfLoad => {
                self.maxes[loc.pos] = self.values.loc_last(&loc).clone_ref(py);
                if self.lengths.is_built() {
                    self.lengths.delete_at(loc.pos);
                }
            }
            ops::Delete::ListsLenGTOne => {
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
                self.maxes[prev] = self.values[prev].last().unwrap().clone_ref(py);
                self.expand(py, prev);
            }
            ops::Delete::LenAtPosNotZero => {
                self.maxes[loc.pos] = self.values.loc_last(&loc).clone_ref(py);
            }
            ops::Delete::Other => self.remove_pos(&loc),
        }
        Ok(())
    }
    fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self.0.values[pos].len(), self.0.load, &self.0.lengths) {
            ops::Expand::LenAtPosGtDoubleLoad => {
                let half = self.0.values[pos].split_off(self.0.load);
                let new_max_at_pos = self.values[pos].last().unwrap().clone_ref(py);
                let last_max = half.last().unwrap().clone_ref(py);
                self.expand_at_pos(pos, half, last_max, new_max_at_pos);
            }
            ops::Expand::IdxBuilt => self.lengths.expand_at(pos),
            ops::Expand::NoOp => (),
        }
    }
    fn extend(&mut self, py: Python<'_>, values: ListAdd<'_, Self>) -> PyResult<()> {
        values.merge_with(
            self,
            py,
            |a, b| a.bind(py).compare_fast(b.bind(py)).unwrap(),
            None,
        )
    }
    fn index(
        &mut self,
        value: &Bound<'_, PyAny>,
        indices: RangeIdxs<Option<isize>>,
    ) -> PyResult<usize> {
        let py = value.py();
        match indices.into_index(self.len) {
            None => Err(errors::not_in_list(value)),
            Some(indices) => {
                let loc = Loc::try_from_index(self, value)?;
                if self.values.loc(&loc).bind(py).ne(value)? {
                    Err(errors::not_in_list(value))
                } else {
                    let left = self.loc(&loc);
                    if indices.start <= left {
                        if left < indices.stop {
                            return Ok(left);
                        }
                    } else if indices.start < self.bisect::<false>(value)? {
                        return Ok(indices.start);
                    }
                    Err(errors::not_in_list(value))
                }
            }
        }
    }
    #[inline]
    fn find(&self, value: &Bound<'_, PyAny>) -> PyResult<Option<Loc>> {
        match ops::Maxes::left(&self.maxes, value) {
            ops::Maxes::BisectErr(err) => Err(err),
            ops::Maxes::Empty | ops::Maxes::LenEQPos(_) => Ok(None),
            ops::Maxes::LenNEPos(mut loc) => {
                loc.idx = self.values[loc.pos].bisect_left(value)?;
                match self.values.loc(&loc).bind(value.py()).eq(value) {
                    Ok(true) => Ok(Some(loc)),
                    Ok(false) => Ok(None),
                    Err(err) => Err(err),
                }
            }
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
