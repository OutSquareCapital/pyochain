use derive_more::{Constructor, From};
use pyo3::{prelude::*, types::slice::PySliceIndices};
use tap::prelude::*;

use crate::{bisect::Bisect, errors, inner::InnerData, traits::ListsMethods};

pub type RawIrangeSpecs<'py> = (
    Python<'py>,
    Option<Bound<'py, PyAny>>,
    Option<Bound<'py, PyAny>>,
    (bool, bool),
);

pub type RawRangeIdxs = (Option<isize>, Option<isize>);
#[derive(Debug, Clone)]
enum RangeBound<'py> {
    None,
    Inclusive(Bound<'py, PyAny>),
    Exclusive(Bound<'py, PyAny>),
}
impl<'py> RangeBound<'py> {
    #[inline]
    fn new(value: Option<Bound<'py, PyAny>>, inclusive: bool) -> Self {
        match (value, inclusive) {
            (None, _) => Self::None,
            (Some(v), true) => Self::Inclusive(v),
            (Some(v), false) => Self::Exclusive(v),
        }
    }
    #[inline]
    fn with_key<'a>(self, key: &'a Bound<'py, PyAny>) -> PyResult<Self> {
        match self {
            Self::None => Ok(Self::None),
            Self::Inclusive(value) => key.call1((value,)).map(Self::Inclusive),
            Self::Exclusive(value) => key.call1((value,)).map(Self::Exclusive),
        }
    }
}
#[derive(Constructor)]
pub(super) struct IrangeSpecs<'py> {
    pub py: Python<'py>,
    min: RangeBound<'py>,
    max: RangeBound<'py>,
}
impl<'py> From<RawIrangeSpecs<'py>> for IrangeSpecs<'py> {
    fn from((py, min, max, (incl_min, incl_max)): RawIrangeSpecs<'py>) -> Self {
        Self {
            py,
            min: RangeBound::new(min, incl_min),
            max: RangeBound::new(max, incl_max),
        }
    }
}
impl IrangeSpecs<'_> {
    pub fn with_keys(values: impl Into<Self>, key: &Py<PyAny>) -> PyResult<Self> {
        let mut slf = values.into();
        let func = key.bind(slf.py);
        slf.min = slf.min.with_key(func)?;
        slf.max = slf.max.with_key(func)?;
        Ok(slf)
    }
    pub fn into_bounds<T: ListsMethods>(self, data: &T) -> Option<PyResult<Bounds>> {
        if data.maxes.is_empty() {
            None
        } else {
            let min = match self.min {
                RangeBound::None => Ok(Loc::default()),
                RangeBound::Inclusive(m) => min_bound::<_, true>(data, &m)?,
                RangeBound::Exclusive(m) => min_bound::<_, false>(data, &m)?,
            };
            let max = match self.max {
                RangeBound::None => Ok(Loc::new_inclusive_max(data)),
                RangeBound::Inclusive(m) => max_bound::<_, false>(data, &m),
                RangeBound::Exclusive(m) => max_bound::<_, true>(data, &m),
            };
            match (min, max) {
                (_, Err(e)) | (Err(e), _) => Some(Err(e)),
                (Ok(min), Ok(max))
                    if min.pos > max.pos || (min.pos == max.pos && min.idx >= max.idx) =>
                {
                    None
                }
                (Ok(min), Ok(max)) => Some(Ok(Bounds::new(min, max))),
            }
        }
    }
}
fn min_bound<T, const LEFT: bool>(data: &T, value: &Bound<'_, PyAny>) -> Option<PyResult<Loc>>
where
    T: ListsMethods,
{
    match data.maxes.bisect::<LEFT>(value) {
        Ok(pos) if pos == data.maxes.len() => None,
        Err(e) => Some(Err(e)),
        Ok(pos) => data.keys()[pos]
            .bisect::<LEFT>(value)
            .map(|idx| Loc::new(pos, idx))
            .pipe(Some),
    }
}
fn max_bound<T, const LEFT: bool>(data: &T, value: &Bound<'_, PyAny>) -> PyResult<Loc>
where
    T: ListsMethods,
{
    let mut pos = data.maxes.bisect::<LEFT>(value)?;
    let idx = if pos == data.maxes.len() {
        pos -= 1;
        data.keys()[pos].len()
    } else {
        data.keys()[pos].bisect::<LEFT>(value)?
    };
    Ok(Loc::new(pos, idx))
}
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
    pub fn into_index(self, data: &InnerData) -> Option<RangeIdxs<usize>> {
        if data.is_empty() {
            None
        } else {
            let length = data.len().cast_signed();
            let mut start = self.start.unwrap_or(0);
            let mut stop = self.stop.unwrap_or(length);
            if start < 0 {
                start += length;
            }
            start = start.max(0);
            if stop < 0 {
                stop += length;
            }
            stop = stop.min(length);
            if stop <= start {
                None
            } else {
                Some(RangeIdxs::new(start.cast_unsigned(), stop.cast_unsigned()))
            }
        }
    }
    pub(super) fn into_bounds(self, data: &mut InnerData) -> Option<PyResult<Bounds>> {
        if data.is_empty() {
            None
        } else {
            let length = data.len();
            let normalize_bound = |mut val: isize| {
                if val < 0 {
                    val += length.cast_signed();
                    val.max(0).cast_unsigned()
                } else {
                    val.cast_unsigned().min(length)
                }
            };
            let start = self.start.map_or(0, normalize_bound);
            let stop = self.stop.map_or(length, normalize_bound);
            if start >= stop {
                None
            } else {
                let min = data.set_pos_usize(start);
                let max = if stop == length {
                    let max_pos = data.values.len() - 1;
                    let max_idx = data.values.last().unwrap().len();
                    Ok(Loc::new(max_pos, max_idx))
                } else {
                    data.set_pos_usize(stop)
                };
                match (min, max) {
                    (Err(e), _) | (_, Err(e)) => Some(Err(e)),
                    (Ok(min), Ok(max)) => Some(Ok(Bounds::new(min, max))),
                }
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
    #[inline]
    fn new_inclusive_max<T: ListsMethods>(data: &T) -> Self {
        let pos = data.maxes.len() - 1;
        let idx = data.keys()[pos].len();
        Self::new(pos, idx)
    }
    pub fn try_from_index<T>(list: &T, value: &Bound<'_, PyAny>) -> PyResult<Loc>
    where
        T: ListsMethods,
    {
        let pos = list.maxes.bisect_left(value)?;
        if pos == list.maxes.len() {
            Err(errors::not_in_list(value))
        } else {
            let idx = list.keys()[pos].bisect_left(value)?;
            Ok(Self::new(pos, idx))
        }
    }
}

#[derive(Debug, PartialEq, Eq, Default, Constructor, Clone, Copy)]
pub struct Bounds {
    pub(super) min: Loc,
    pub(super) max: Loc,
}
