use derive_more::{Constructor, From};
use pyo3::{prelude::*, types::slice::PySliceIndices};

use crate::inner::InnerList;

pub type RawIrangeSpecs<'py> = (
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    (bool, bool),
);

#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor, Debug, From)]
pub(super) struct RangeIdxs<T> {
    pub start: T,
    pub stop: T,
}
impl From<PySliceIndices> for RangeIdxs<usize> {
    fn from(i: PySliceIndices) -> Self {
        if i.step > 0 {
            (i.start.cast_unsigned(), i.stop.cast_unsigned())
        } else {
            ((i.stop + 1).cast_unsigned(), (i.start + 1).cast_unsigned())
        }
        .into()
    }
}
impl IntoIterator for RangeIdxs<usize> {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.start..self.stop
    }
}
impl RangeIdxs<Option<isize>> {
    #[must_use]
    pub fn into_index(self, list: &InnerList) -> Option<RangeIdxs<usize>> {
        if list.is_empty() {
            None
        } else {
            let length = list.len();
            let start = self.start.map_or_default(|mut s| {
                if s.is_negative() {
                    s += length.cast_signed();
                }
                s.max(0).cast_unsigned()
            });
            let stop = self.stop.map_or(length, |mut s| {
                if s.is_negative() {
                    s += length.cast_signed();
                }
                s.cast_unsigned().min(length)
            });
            if stop <= start {
                None
            } else {
                Some(RangeIdxs::new(start, stop))
            }
        }
    }
}
#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor, Debug)]
pub struct Loc {
    pub pos: usize,
    pub idx: usize,
}
impl Loc {
    #[must_use]
    pub const fn with_idx(idx: usize) -> Self {
        Self::new(0, idx)
    }
    #[must_use]
    pub const fn with_pos(pos: usize) -> Self {
        Self::new(pos, 0)
    }
    pub(super) fn into_index(self, list: &mut InnerList) -> usize {
        if self.pos == 0 {
            self.idx
        } else {
            if list.lengths.is_not_built() {
                list.lengths.build(&list.values);
            }
            list.lengths.loc_at(&self)
        }
    }
}
