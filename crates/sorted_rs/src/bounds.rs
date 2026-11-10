use derive_more::{Constructor, From};
use pyo3::{prelude::*, types::slice::PySliceIndices};

use crate::inner::InnerList;

pub type RawIrangeSpecs<'py> = (
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    (bool, bool),
);

#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor, Debug, From)]
pub(super) struct Bounds<T> {
    pub start: T,
    pub stop: T,
}
impl From<PySliceIndices> for Bounds<usize> {
    fn from(i: PySliceIndices) -> Self {
        if i.step > 0 {
            (i.start.cast_unsigned(), i.stop.cast_unsigned())
        } else {
            ((i.stop + 1).cast_unsigned(), (i.start + 1).cast_unsigned())
        }
        .into()
    }
}
impl IntoIterator for Bounds<usize> {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;
    fn into_iter(self) -> Self::IntoIter {
        self.start..self.stop
    }
}
impl Bounds<Option<isize>> {
    #[must_use]
    pub fn try_normalize(self, list: &InnerList) -> Option<Bounds<usize>> {
        if list.is_empty() {
            None
        } else {
            let length = list.len().cast_signed();
            let start = match self.start {
                None => 0,
                Some(s) if s.is_negative() => (s + length).max(0),
                Some(s) => s,
            };
            let stop = match self.stop {
                None => length,
                Some(s) if s.is_negative() => s + length,
                Some(s) => s.min(length),
            };
            if start >= stop {
                None
            } else {
                Some(Bounds::new(start.cast_unsigned(), stop.cast_unsigned()))
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
