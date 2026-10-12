use derive_more::Constructor;
use pyo3::prelude::*;

use crate::{InnerData, bisect::Bisect, types::VecPy};

#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor, Debug)]
pub struct Loc {
    pub pos: usize,
    pub idx: usize,
}
impl Loc {
    #[must_use]
    pub fn with_idx(idx: usize) -> Self {
        Self::new(0, idx)
    }
    #[must_use]
    pub fn with_pos(pos: usize) -> Self {
        Self::new(pos, 0)
    }
}

#[derive(PartialEq, Eq, Default, Constructor, Clone, Copy)]
pub struct Bounds {
    pub min: Loc,
    pub max: Loc,
}
impl Bounds {
    pub fn from_irange(
        lists: &[VecPy],
        maxes: &[Py<PyAny>],
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Bounds> {
        if maxes.is_empty() {
            Ok(Self::default())
        } else {
            let min = match minimum {
                None => Loc::default(),
                Some(minimum) => {
                    if inclusive.0 {
                        let min_pos = maxes.bisect_left(&minimum)?;

                        if min_pos == maxes.len() {
                            return Ok(Self::default());
                        }
                        let min_idx = lists[min_pos].bisect_left(&minimum)?;
                        Loc::new(min_pos, min_idx)
                    } else {
                        let min_pos = maxes.bisect_right(&minimum)?;
                        if min_pos == maxes.len() {
                            return Ok(Self::default());
                        }
                        let min_idx = lists[min_pos].bisect_right(&minimum)?;
                        Loc::new(min_pos, min_idx)
                    }
                }
            };

            // Calculate the maximum (pos, idx) pair. By default this location
            // will be exclusive in our calculation.
            let max = maximum.map_or_else(
                || {
                    let max_pos = maxes.len() - 1;
                    let max_idx = lists[max_pos].len();
                    Ok(Loc::new(max_pos, max_idx))
                },
                |m| {
                    if inclusive.1 {
                        let mut max_pos = maxes.bisect_right(&m)?;

                        let max_idx = if max_pos == maxes.len() {
                            max_pos -= 1;
                            lists[max_pos].len()
                        } else {
                            lists[max_pos].bisect_right(&m)?
                        };
                        Ok::<_, PyErr>(Loc::new(max_pos, max_idx))
                    } else {
                        let mut max_pos = maxes.bisect_left(&m)?;

                        let max_idx = if max_pos == maxes.len() {
                            max_pos -= 1;
                            lists[max_pos].len()
                        } else {
                            lists[max_pos].bisect_left(&m)?
                        };
                        Ok(Loc::new(max_pos, max_idx))
                    }
                },
            )?;

            if min.pos > max.pos || (min.pos == max.pos && min.idx >= max.idx) {
                Ok(Self::default())
            } else {
                Ok(Self::new(min, max))
            }
        }
    }
    pub fn from_islice(
        data: &mut InnerData,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Self> {
        let length = data.len.cast_signed();

        if length == 0 {
            Ok(Self::default())
        } else {
            let normalize_bound = |mut val: isize| {
                if val < 0 {
                    val += length;
                    val.max(0)
                } else {
                    val.min(length)
                }
            };
            let start = start.map_or(0, normalize_bound);
            let stop = stop.map_or(length, normalize_bound);
            if start >= stop {
                Ok(Self::default())
            } else {
                let mut bounds = Self::default();
                data.set_pos(start, &mut bounds.min)?;

                if stop == length {
                    bounds.max.pos = data.values.len() - 1;
                    bounds.max.idx = data.values.last().unwrap().len();
                } else {
                    data.set_pos(stop, &mut bounds.max)?;
                }
                Ok(bounds)
            }
        }
    }
}
