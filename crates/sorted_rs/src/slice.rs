use itertools::Itertools;
use pyo3::{prelude::*, types::PySlice};
use tap::prelude::*;

use crate::{
    Bounds, InnerData, Loc, ops,
    traits::{ListsDataMethods, NestedVec},
    types::VecPy,
};

#[derive(Clone, Copy)]
pub(super) struct SliceIdxs<'py> {
    py: Python<'py>,
    start: isize,
    stop: isize,
    step: isize,
    /// Length of the referenced sequence.
    ref_len: isize,
}
impl<'py> SliceIdxs<'py> {
    pub fn new(slice: &Bound<'py, PySlice>, length: usize) -> PyResult<Self> {
        let ref_len = length.cast_signed();
        let i = slice.indices(ref_len)?;
        Self {
            py: slice.py(),
            start: i.start,
            stop: i.stop,
            step: i.step,
            ref_len,
        }
        .pipe(Ok)
    }
    fn kind(&self) -> ops::Slice {
        ops::Slice::new(self.start, self.stop, self.step, self.ref_len)
    }
    pub fn extract_from(&self, list: &mut InnerData) -> PyResult<VecPy> {
        let py = self.py;
        match self.kind() {
            ops::Slice::NoOp => Ok(Vec::new()),
            ops::Slice::Full => list.collapse(py).pipe(Ok),
            ops::Slice::Forward | ops::Slice::ForwardSmallToAdjust => {
                forward_extract::<false>(self, list)
            }
            ops::Slice::ForwardToEnd | ops::Slice::ForwardSmall => {
                forward_extract::<true>(self, list)
            }
            ops::Slice::ForwardStep | ops::Slice::MayReverse => self
                .iter()
                .map(|i| list.get_item(py, i))
                .map_ok(Bound::unbind)
                .collect(),
            ops::Slice::Backward => {
                let mut result = self
                    .to_new_with(self.stop + 1, self.start + 1)
                    .extract_from(list)?;
                result.reverse();
                Ok(result)
            }
            ops::Slice::BackwardStep => self
                .iter()
                .rev()
                .map(|i| list.get_item(py, i))
                .map_ok(Bound::unbind)
                .collect(),
        }
    }

    pub fn delete_from<T: ListsDataMethods>(&self, list: &mut T) -> PyResult<()> {
        let py = self.py;
        match self.kind() {
            ops::Slice::NoOp => Ok(()),
            ops::Slice::Full => {
                list.clear(py);
                Ok(())
            }
            ops::Slice::ForwardSmallToAdjust => forward_delete_small::<_, true>(self, list),
            ops::Slice::ForwardSmall => forward_delete_small::<_, false>(self, list),
            _ => {
                let mut loc = Loc::default();
                self.iter().rev().try_for_each(|idx| {
                    list.set_pos(idx, &mut loc)?;
                    list.delete(py, &mut loc)
                })
            }
        }
    }
    fn to_new_with(self, start: isize, stop: isize) -> Self {
        Self {
            py: self.py,
            start,
            stop,
            step: 1,
            ref_len: self.ref_len,
        }
    }
    #[allow(clippy::range_plus_one)]
    fn iter(self) -> std::iter::StepBy<std::ops::Range<isize>> {
        if self.step.is_positive() {
            (self.start..self.stop).step_by(self.step.cast_unsigned())
        } else {
            let step: isize = self.step.abs();
            let lo = self.start - (self.start - self.stop - 1) / step * step;
            (lo..self.start + 1).step_by(step.cast_unsigned())
        }
    }
}
fn forward_extract<const TO_END: bool>(slice: &SliceIdxs, list: &mut InnerData) -> PyResult<VecPy> {
    let py = slice.py;
    let mut bounds = Bounds::default();
    list.set_pos(slice.start, &mut bounds.min)?;
    let start_list = &list.values[bounds.min.pos];
    bounds.max.idx = bounds.min.idx + (slice.stop - slice.start).cast_unsigned();
    match (start_list.len() >= bounds.max.idx, TO_END) {
        // Small slice optimization: start index and stop index are within the start list.
        (true, _) => start_list[bounds.min.idx..bounds.max.idx]
            .iter()
            .map(|x| x.clone_ref(py))
            .collect(),
        (false, true) => {
            bounds.max.pos = list.values.len() - 1;
            bounds.max.idx = list.values.loc_len(&bounds.max);
            get_slice(py, &list.values, &bounds)
        }
        (false, false) => {
            list.set_pos(slice.stop, &mut bounds.max)?;
            get_slice(py, &list.values, &bounds)
        }
    }
    .pipe(Ok)
}

fn get_slice(py: Python<'_>, lists: &[VecPy], bounds: &Bounds) -> VecPy {
    lists[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(lists[bounds.min.pos + 1..bounds.max.pos].iter().flatten())
        .chain(lists[bounds.max.pos][0..bounds.max.idx].iter())
        .map(|x| x.clone_ref(py))
        .collect()
}
fn forward_delete_small<T, const NEED_ADJUST: bool>(slice: &SliceIdxs, list: &mut T) -> PyResult<()>
where
    T: ListsDataMethods,
{
    let py = slice.py;
    let mut values = slice.to_new_with(0, slice.start).extract_from(list)?;
    if NEED_ADJUST {
        let mut new_slice = slice
            .to_new_with(slice.stop, slice.ref_len)
            .extract_from(list)?;
        values.append(new_slice.as_mut());
    }
    list.clear(py);
    list.extend(py, values.into())
}
