use pyo3::{prelude::*, types::PySlice};
use tap::Pipe;

use crate::{bisect, data::ListsData, errors};

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
    pub fn set_from_pos(&mut self, mut idx: isize, data: &mut ListsData) -> PyResult<()> {
        if idx < 0 {
            if idx >= -data.lists.last().unwrap().len().cast_signed() {
                self.pos = data.lists.len() - 1;
                self.idx = (data.lists.last().unwrap().len().cast_signed() + idx).cast_unsigned();
                return Ok(());
            }

            idx += data.len.cast_signed();

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= data.len.cast_signed() {
            return errors::out_of_range_err();
        }

        if idx < data.lists[0].len().cast_signed() {
            self.pos = 0;
            self.idx = idx.cast_unsigned();
            return Ok(());
        }

        if data.idx.is_empty() {
            data.build_index();
        }
        let pos = data.idx.pipe_ref_mut(|index| {
            let mut pos = 0;
            let mut child = 1;
            let len_index = index.len();

            while child < len_index {
                let index_child = index[child].cast_signed();

                if idx < index_child {
                    pos = child;
                } else {
                    idx -= index_child;
                    pos = child + 1;
                }

                child = (pos << 1) + 1;
            }
            pos
        });

        self.pos = pos - data.offset;
        self.idx = idx.cast_unsigned();
        Ok(())
    }

    pub fn loc(&self, data: &mut ListsData) -> PyResult<isize> {
        if self.pos == 0 {
            Ok(self.idx.cast_signed())
        } else {
            if data.idx.is_empty() {
                data.build_index();
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            let mut pos = self.pos + data.offset;
            // Iterate until reaching the root of the index tree at pos = 0.
            let total = data.idx.pipe_ref_mut(|idx| {
                let mut total = 0;
                while pos != 0 {
                    // Right-child nodes are at even indices. At such indices
                    // account the total below the left child node.

                    if pos.is_multiple_of(2) {
                        total += idx[pos - 1].cast_signed();
                    }

                    // Advance pos to the parent node.

                    pos = (pos - 1) >> 1;
                }
                total
            });

            Ok(total + self.idx.cast_signed())
        }
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
    pub fn get_irange_specs(
        lists: &[Vec<Py<PyAny>>],
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

    pub fn get_islice_specs(
        data: &mut ListsData,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<Bounds>> {
        let length = data.len.cast_signed();
        let mut bounds = Bounds::default();

        if length == 0 {
            Ok(None)
        } else {
            //NOTE: Need to investiguate why we need to use PySlice at all. Same pattern in SliceView original code.
            let indices =
                PySlice::new(py, start.unwrap_or(0), stop.unwrap_or(length), 1).indices(length)?;

            if indices.start >= indices.stop {
                Ok(None)
            } else {
                bounds.min.set_from_pos(indices.start, data)?;

                if indices.stop == length {
                    bounds.max.pos = data.lists.len() - 1;
                    bounds.max.idx = data.lists.last().unwrap().len();
                } else {
                    bounds.max.set_from_pos(indices.stop, data)?;
                }

                Ok(Some(bounds))
            }
        }
    }
}
