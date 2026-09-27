use derive_more::Constructor;
use pyo3::prelude::*;

use crate::{bisect::Bisect, types::VecPy};

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
