use derive_more::{Deref, From};
use itertools::Itertools;
use pyo3::{prelude::*, types::PySlice};
use tap::prelude::*;

use crate::{
    bounds::{Loc, RangeIdxs},
    inner::InnerData,
    ops::{self, ForwardSlice},
    traits::ListsMethods,
    types::VecPy,
};

#[derive(Clone, Copy, Deref)]
pub(super) struct SliceIdxs {
    #[deref]
    indices: RangeIdxs<usize>,
    step: ops::Nb,
    /// Length of the referenced sequence.
    ref_len: usize,
}
impl SliceIdxs {
    pub fn new(slice: &Bound<'_, PySlice>, ref_len: usize) -> PyResult<Self> {
        let i = slice.indices(ref_len.cast_signed())?;
        let step = i.step.into();
        Self {
            indices: i.into(),
            step,
            ref_len,
        }
        .pipe(Ok)
    }
    fn kind(&self) -> ops::Slice {
        ops::Slice::new(self.indices, self.step, self.ref_len)
    }
    pub fn extract_from(&self, py: Python<'_>, list: &mut InnerData) -> PyResult<VecPy> {
        match self.kind() {
            ops::Slice::NoOp => Ok(Vec::new()),
            ops::Slice::Forward(fwd) => self.extract_fwd(py, fwd, list),
            ops::Slice::ForwardStep(s) => self
                .into_iter()
                .step_by(s)
                .conv::<ItemsVisitor<_>>()
                .get_all(list, py),
            ops::Slice::Backward => {
                let fwd = ops::ForwardSlice::new(self.indices, self.ref_len);
                self.extract_fwd(py, fwd, list)?
                    .tap_mut(|r| r.reverse())
                    .pipe(Ok)
            }
            ops::Slice::BackwardStep(s) => self
                .into_iter()
                .rev()
                .step_by(s)
                .conv::<ItemsVisitor<_>>()
                .get_all(list, py),
        }
    }

    pub fn delete_from<T: ListsMethods>(&self, py: Python<'_>, list: &mut T) -> PyResult<()> {
        match self.kind() {
            ops::Slice::NoOp => Ok(()),
            ops::Slice::Forward(fwd) => match fwd {
                ForwardSlice::Full => {
                    list.clear(py);
                    Ok(())
                }
                ForwardSlice::SmallToAdjust => self.delete_fwd_small::<_, true>(py, list),
                ForwardSlice::Small => self.delete_fwd_small::<_, false>(py, list),
                ForwardSlice::ToEnd | ForwardSlice::Partial => self
                    .into_iter()
                    .rev()
                    .conv::<ItemsVisitor<_>>()
                    .del_all(list, py),
            },
            ops::Slice::Backward => self
                .into_iter()
                .rev()
                .conv::<ItemsVisitor<_>>()
                .del_all(list, py),
            ops::Slice::ForwardStep(s) => self
                .into_iter()
                .step_by(s)
                .rev()
                .conv::<ItemsVisitor<_>>()
                .del_all(list, py),
            ops::Slice::BackwardStep(s) => self
                .into_iter()
                .rev()
                .step_by(s)
                .conv::<ItemsVisitor<_>>()
                .del_all(list, py),
        }
    }
    fn extract_fwd(
        &self,
        py: Python<'_>,
        fwd: ForwardSlice,
        list: &mut InnerData,
    ) -> PyResult<VecPy> {
        fn fwd_extract<const TO_END: bool>(
            slf: &SliceIdxs,
            py: Python<'_>,
            list: &mut InnerData,
        ) -> PyResult<VecPy> {
            let min = list.set_pos_usize(slf.start)?;
            let mut max = Loc::with_idx(min.idx + (slf.stop - slf.start));
            let start_list = &list.values[min.pos];
            match (start_list.len() >= max.idx, TO_END) {
                // Small slice optimization: start index and stop index are within the start list.
                (true, _) => start_list[min.idx..max.idx]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect(),
                (false, true) => {
                    max.pos = list.values.len() - 1;
                    max.idx = list.values.loc_len(&max);
                    get_slice(py, &list.values, min, max)
                }
                (false, false) => {
                    max = list.set_pos_usize(slf.stop)?;
                    get_slice(py, &list.values, min, max)
                }
            }
            .pipe(Ok)
        }
        match fwd {
            ForwardSlice::Full => list.values.iter_owned(py).collect::<Vec<_>>().pipe(Ok),
            ForwardSlice::Partial | ForwardSlice::SmallToAdjust => {
                fwd_extract::<false>(self, py, list)
            }
            ForwardSlice::ToEnd | ForwardSlice::Small => fwd_extract::<true>(self, py, list),
        }
    }

    fn delete_fwd_small<T, const NEED_ADJUST: bool>(
        &self,
        py: Python<'_>,
        list: &mut T,
    ) -> PyResult<()>
    where
        T: ListsMethods,
    {
        let mut values = reset_slice((0, self.start), self.ref_len).extract_from(py, list)?;
        if NEED_ADJUST {
            let mut new_slice =
                reset_slice((self.stop, self.ref_len), self.ref_len).extract_from(py, list)?;
            values.append(new_slice.as_mut());
        }
        list.clear(py);
        list.extend(py, values.into())
    }
}
#[derive(From)]
struct ItemsVisitor<T>(T);
impl<T> ItemsVisitor<T>
where
    T: DoubleEndedIterator<Item = usize>,
{
    fn del_all<L: ListsMethods>(mut self, list: &mut L, py: Python<'_>) -> PyResult<()> {
        self.0.try_for_each(|idx| {
            let loc = list.set_pos_usize(idx)?;
            list.delete(py, loc)
        })
    }
    fn get_all(self, list: &mut InnerData, py: Python<'_>) -> PyResult<VecPy> {
        self.0
            .map(usize::cast_signed)
            .map(|i| list.get_item(py, i))
            .map_ok(Bound::unbind)
            .collect()
    }
}
fn get_slice(py: Python<'_>, lists: &[VecPy], min: Loc, max: Loc) -> VecPy {
    lists[min.pos][min.idx..]
        .iter()
        .chain(lists[min.pos + 1..max.pos].iter().flatten())
        .chain(lists[max.pos][0..max.idx].iter())
        .map(|x| x.clone_ref(py))
        .collect()
}
fn reset_slice<T: Into<RangeIdxs<usize>>>(indices: T, ref_len: usize) -> SliceIdxs {
    SliceIdxs {
        indices: indices.into(),
        step: ops::Nb::One,
        ref_len,
    }
}
