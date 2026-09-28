use std::sync::atomic::{AtomicUsize, Ordering};

use derive_more::Constructor;
use pyo3::prelude::*;

use crate::{InnerData, bisect::Bisect, types::VecPy};

#[derive(PartialEq, Eq, Default, Clone, Copy, Constructor)]
pub struct Loc {
    pub pos: usize,
    pub idx: usize,
}
impl Loc {
    #[must_use]
    pub fn with_idx(idx: usize) -> Self {
        Self { pos: 0, idx }
    }
    #[must_use]
    pub fn with_pos(pos: usize) -> Self {
        Self { pos, idx: 0 }
    }
}

#[derive(Default, Constructor)]
pub struct Bounds {
    pub min: Loc,
    pub max: Loc,
}
impl Bounds {
    #[must_use]
    pub fn from_full_iter(data: &InnerData) -> Self {
        let max = data
            .values
            .last()
            .map_or(Loc::default(), |v| Loc::new(data.values.len() - 1, v.len()));
        Self::new(Loc::default(), max)
    }
    pub fn from_sorted(
        lists: &[VecPy],
        maxes: &[Py<PyAny>],
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        if maxes.is_empty() {
            Ok(None)
        } else {
            let min = match minimum {
                None => Loc::default(),
                Some(minimum) => {
                    if inclusive.0 {
                        let min_pos = maxes.bisect_left(&minimum)?;

                        if min_pos == maxes.len() {
                            return Ok(None);
                        }
                        let min_idx = lists[min_pos].bisect_left(&minimum)?;
                        Loc::new(min_pos, min_idx)
                    } else {
                        let min_pos = maxes.bisect_right(&minimum)?;
                        if min_pos == maxes.len() {
                            return Ok(None);
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
                Ok(None)
            } else {
                Ok(Some(Bounds { min, max }))
            }
        }
    }
}
#[derive(Constructor, Debug)]
pub struct AtomicLoc {
    pub(super) pos: AtomicUsize,
    pub(super) idx: AtomicUsize,
}
impl AtomicLoc {
    #[inline]
    pub fn load(&self) -> (usize, usize) {
        (
            self.pos.load(Ordering::Relaxed),
            self.idx.load(Ordering::Relaxed),
        )
    }
    pub fn store(&self, pos: usize, idx: usize) {
        self.pos.store(pos, Ordering::Relaxed);
        self.idx.store(idx, Ordering::Relaxed);
    }
}
impl From<Loc> for AtomicLoc {
    fn from(loc: Loc) -> Self {
        Self::new(loc.pos.into(), loc.idx.into())
    }
}
#[derive(Constructor, Debug)]
pub struct AtomicBounds {
    pub(super) min: AtomicLoc,
    pub(super) max: AtomicLoc,
}
impl From<Bounds> for AtomicBounds {
    fn from(bounds: Bounds) -> Self {
        Self::new(bounds.min.into(), bounds.max.into())
    }
}
