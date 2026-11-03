use derive_more::Constructor;
use pyo3::{prelude::*, types::slice::PySliceIndices};
use tap::Pipe;

use crate::{InnerData, bisect::Bisect, errors, traits::ListsDataMethods};
#[derive(Constructor)]
pub struct IrangeSpecs<'py> {
    pub py: Python<'py>,
    min: Option<Bound<'py, PyAny>>,
    max: Option<Bound<'py, PyAny>>,
    inclusive: (bool, bool),
}
impl<'py> IrangeSpecs<'py> {
    pub fn with_keys(self, key: &Bound<'py, PyAny>) -> PyResult<Self> {
        let min = self.min.map(|value| key.call1((value,))).transpose()?;
        let max = self.max.map(|value| key.call1((value,))).transpose()?;
        Self::new(self.py, min, max, self.inclusive).pipe(Ok)
    }
    pub fn into_bounds<T: ListsDataMethods>(self, data: &T) -> PyResult<Bounds> {
        if data.maxes.is_empty() {
            Ok(Bounds::default())
        } else {
            let min = match self.min {
                None => Loc::default(),
                Some(minimum) => {
                    if self.inclusive.0 {
                        let min_pos = data.maxes.bisect_left(&minimum)?;
                        if min_pos == data.maxes.len() {
                            return Ok(Bounds::default());
                        }
                        let min_idx = data.keys()[min_pos].bisect_left(&minimum)?;
                        Loc::new(min_pos, min_idx)
                    } else {
                        let min_pos = data.maxes.bisect_right(&minimum)?;
                        if min_pos == data.maxes.len() {
                            return Ok(Bounds::default());
                        }
                        let min_idx = data.keys()[min_pos].bisect_right(&minimum)?;
                        Loc::new(min_pos, min_idx)
                    }
                }
            };

            // Calculate the maximum (pos, idx) pair. By default this location
            // will be exclusive in our calculation.
            let max = self.max.map_or_else(
                || {
                    let max_pos = data.maxes.len() - 1;
                    let max_idx = data.keys()[max_pos].len();
                    Ok(Loc::new(max_pos, max_idx))
                },
                |m| {
                    if self.inclusive.1 {
                        let mut max_pos = data.maxes.bisect_right(&m)?;
                        let max_idx = if max_pos == data.maxes.len() {
                            max_pos -= 1;
                            data.keys()[max_pos].len()
                        } else {
                            data.keys()[max_pos].bisect_right(&m)?
                        };
                        Ok::<_, PyErr>(Loc::new(max_pos, max_idx))
                    } else {
                        let mut max_pos = data.maxes.bisect_left(&m)?;

                        let max_idx = if max_pos == data.maxes.len() {
                            max_pos -= 1;
                            data.keys()[max_pos].len()
                        } else {
                            data.keys()[max_pos].bisect_left(&m)?
                        };
                        Ok(Loc::new(max_pos, max_idx))
                    }
                },
            )?;

            if min.pos > max.pos || (min.pos == max.pos && min.idx >= max.idx) {
                Ok(Bounds::default())
            } else {
                Ok(Bounds::new(min, max))
            }
        }
    }
}
#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor, Debug)]
pub struct RangeIdxs<T> {
    pub start: T,
    pub stop: T,
}
impl<T> From<(T, T)> for RangeIdxs<T> {
    fn from(value: (T, T)) -> Self {
        Self::new(value.0, value.1)
    }
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
    pub fn into_index(self, length: usize) -> Option<RangeIdxs<usize>> {
        let length = length.cast_signed();
        if length == 0 {
            None
        } else {
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
    pub fn into_bounds(self, data: &mut InnerData) -> PyResult<Bounds> {
        let length = data.len;
        if length == 0 {
            Ok(Bounds::default())
        } else {
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
                Ok(Bounds::default())
            } else {
                let min = data.set_pos_usize(start)?;
                let max = if stop == length {
                    let max_pos = data.values.len() - 1;
                    let max_idx = data.values.last().unwrap().len();
                    Loc::new(max_pos, max_idx)
                } else {
                    data.set_pos_usize(stop)?
                };
                Ok(Bounds::new(min, max))
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
    pub fn try_from_index<T>(list: &T, value: &Bound<'_, PyAny>) -> PyResult<Loc>
    where
        T: ListsDataMethods,
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

#[derive(PartialEq, Eq, Default, Constructor, Clone, Copy)]
pub struct Bounds {
    pub min: Loc,
    pub max: Loc,
}
