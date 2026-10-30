use derive_more::Constructor;
use pyo3::{prelude::*, types::PySlice};
use tap::prelude::*;

#[derive(Clone, Copy, Constructor)]
pub struct SliceIdxs<'py> {
    pub py: Python<'py>,
    pub start: isize,
    pub stop: isize,
    pub step: isize,
}
impl<'py> SliceIdxs<'py> {
    pub fn from_pyslice(slice: &Bound<'py, PySlice>, length: isize) -> PyResult<Self> {
        let indices = slice.indices(length)?;
        Self::new(slice.py(), indices.start, indices.stop, indices.step).pipe(Ok)
    }
    pub fn iter(&self) -> impl Iterator<Item = isize> + '_ {
        std::iter::successors(Some(self.start), move |&i| {
            (i + self.step > self.stop).then_some(i + self.step)
        })
    }
    pub fn into_range(self) -> std::iter::StepBy<std::ops::Range<isize>> {
        (self.start..self.stop).step_by(self.step.cast_unsigned())
    }
}
