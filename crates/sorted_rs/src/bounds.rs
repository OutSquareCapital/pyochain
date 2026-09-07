use pyo3::prelude::*;

use crate::{bisect, inner::VecPy};

pub struct Indexes {
    pub start: isize,
    pub stop: isize,
}
impl Indexes {
    #[inline]
    #[must_use]
    pub fn new(start: Option<isize>, stop: Option<isize>, length: isize) -> Self {
        let mut start = start.unwrap_or(0);
        let mut stop = stop.unwrap_or(length);
        if start < 0 {
            start += length;
        }
        start = start.max(0);
        if stop < 0 {
            stop += length;
        }
        stop = stop.min(length);
        Self { start, stop }
    }
}

#[derive(PartialEq, Eq, Default)]
pub struct Pos {
    pub pos: usize,
    pub idx: usize,
}
impl Pos {
    #[must_use]
    pub fn new(pos: usize, idx: usize) -> Self {
        Self { pos, idx }
    }
}

#[derive(Default)]
pub struct Bounds {
    pub min: Pos,
    pub max: Pos,
}
impl Bounds {
    #[must_use]
    pub fn new(min_pos: usize, min_idx: usize, max_pos: usize, max_idx: usize) -> Self {
        Self {
            min: Pos::new(min_pos, min_idx),
            max: Pos::new(max_pos, max_idx),
        }
    }
    pub fn from_sorted(
        lists: &[VecPy],
        maxes: &[Py<PyAny>],
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        if maxes.is_empty() {
            return Ok(None);
        }

        let min = match minimum {
            None => Pos::default(),
            Some(minimum) => {
                if inclusive.0 {
                    let min_pos = bisect::left(maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::left(&lists[min_pos], &minimum)?;
                    Pos::new(min_pos, min_idx)
                } else {
                    let min_pos = bisect::right(maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::right(&lists[min_pos], &minimum)?;
                    Pos::new(min_pos, min_idx)
                }
            }
        };

        // Calculate the maximum (pos, idx) pair. By default this location
        // will be exclusive in our calculation.
        let max = maximum.map_or_else(
            || {
                let max_pos = maxes.len() - 1;
                let max_idx = lists[max_pos].len();
                Ok(Pos::new(max_pos, max_idx))
            },
            |m| {
                if inclusive.1 {
                    let mut max_pos = bisect::right(maxes, &m)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        lists[max_pos].len()
                    } else {
                        bisect::right(&lists[max_pos], &m)?
                    };
                    Ok::<_, PyErr>(Pos::new(max_pos, max_idx))
                } else {
                    let mut max_pos = bisect::left(maxes, &m)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        lists[max_pos].len()
                    } else {
                        bisect::left(&lists[max_pos], &m)?
                    };
                    Ok(Pos::new(max_pos, max_idx))
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
