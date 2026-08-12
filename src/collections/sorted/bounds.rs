use pyo3::{prelude::*, types::PySlice};
use tap::Pipe;

use crate::collections::sorted::{bisect, data::ListsData, errors};

#[derive(PartialEq, Eq, Default)]
pub(super) struct Pos {
    pub pos: usize,
    pub idx: usize,
}
impl Pos {
    pub fn new(pos: usize, idx: usize) -> Self {
        Self { pos, idx }
    }
    pub(super) fn set_from_pos(&mut self, mut idx: isize, data: &mut ListsData) -> PyResult<()> {
        if idx < 0 {
            if idx >= -(data.lists.last().unwrap().len() as isize) {
                self.pos = data.lists.len() - 1;
                self.idx = (data.lists.last().unwrap().len() as isize + idx) as usize;
                return Ok(());
            }

            idx += data.len as isize;

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= data.len as isize {
            return errors::out_of_range_err();
        }

        if idx < data.lists[0].len() as isize {
            self.pos = 0;
            self.idx = idx as usize;
            return Ok(());
        }

        if data.idx.is_empty() {
            data.build_index()?;
        }
        let pos = data.idx.pipe_ref_mut(|index| {
            let mut pos = 0;
            let mut child = 1;
            let len_index = index.len();

            while child < len_index {
                let index_child = index[child] as isize;

                if idx < index_child {
                    pos = child;
                } else {
                    idx -= index_child;
                    pos = child + 1;
                }

                child = (pos << 1) + 1
            }
            pos
        });

        self.pos = pos - data.offset;
        self.idx = idx as usize;
        Ok(())
    }

    ///Convert an index pair (lists index, sublist index) into a single index number.

    ///This number corresponds to the position of the value in the sorted list.

    ///Many queries require the index be built.
    /// Details of the index are described in ``SortedList._build_index``.

    ///Indexing requires traversing the tree from a leaf node to the root.
    ///The parent of each node is easily computable at ``(pos - 1) // 2``.

    ///Left-child nodes are always at odd indices and right-child nodes are always at even indices.

    ///When traversing up from a right-child node, increment the total by the left-child node.

    ///The final index is the sum from traversal and the index in the sublist.

    ///For example, using the index from `SortedList._build_index`:

    ///    _index = 14 5 9 3 2 4 5
    ///    _offset = 3

    ///Tree::

    ///            14
    ///        5      9
    ///    3   2  4   5

    ///Converting an index pair (2, 3) into a single index involves iterating like so:

    ///1. Starting at the leaf node: offset + alpha = 3 + 2 = 5. We identify
    ///    the node as a left-child node. At such nodes, we simply traverse to
    ///    the parent.

    ///2. At node 9, position 2, we recognize the node as a right-child node
    ///    and accumulate the left-child in our total. Total is now 5 and we
    ///    traverse to the parent at position 0.

    ///3. Iteration ends at the root.

    ///The index is then the sum of the total and sublist index: 5 + 3 = 8.
    pub(crate) fn loc(&self, data: &mut ListsData) -> PyResult<isize> {
        if self.pos == 0 {
            Ok(self.idx as isize)
        } else {
            if data.idx.is_empty() {
                data.build_index()?;
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            let mut pos = self.pos + data.offset;
            // Iterate until reaching the root of the index tree at pos = 0.
            let total = data.idx.pipe_ref_mut(|idx| {
                let mut total = 0;
                while pos != 0 {
                    // Right-child nodes are at even indices. At such indices
                    // account the total below the left child node.

                    if pos % 2 == 0 {
                        total += idx[pos - 1] as isize;
                    }

                    // Advance pos to the parent node.

                    pos = (pos - 1) >> 1;
                }
                total
            });

            Ok(total + self.idx as isize)
        }
    }
}

#[derive(Default)]
pub(super) struct Bounds {
    pub min: Pos,
    pub max: Pos,
}
impl Bounds {
    pub fn new(min_pos: usize, min_idx: usize, max_pos: usize, max_idx: usize) -> Self {
        Self {
            min: Pos::new(min_pos, min_idx),
            max: Pos::new(max_pos, max_idx),
        }
    }
    pub(super) fn get_irange_specs(
        lists: &Vec<Vec<Py<PyAny>>>,
        maxes: &Vec<Py<PyAny>>,
        minimum: Option<Bound<'_, PyAny>>,
        maximum: Option<Bound<'_, PyAny>>,
        inclusive: (bool, bool),
    ) -> PyResult<Option<Bounds>> {
        if maxes.is_empty() {
            return Ok(None);
        }
        let mut bounds = Bounds::default();

        // Calculate the minimum (pos, idx) pair. By default this location
        // will be inclusive in our calculation.
        bounds.min = match minimum {
            None => Pos::default(),
            Some(minimum) => {
                if inclusive.0 {
                    let min_pos = bisect::left(&maxes, &minimum)?;

                    if min_pos == maxes.len() {
                        return Ok(None);
                    }

                    let min_idx = bisect::left(&lists[min_pos], &minimum)?;
                    Pos::new(min_pos, min_idx)
                } else {
                    let min_pos = bisect::right(&maxes, &minimum)?;

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
        bounds.max = maximum
            .map(|m| {
                if inclusive.1 {
                    let mut max_pos = bisect::right(&maxes, &m)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        lists[max_pos].len()
                    } else {
                        bisect::right(&lists[max_pos], &m)?
                    };
                    Ok::<_, PyErr>(Pos::new(max_pos, max_idx))
                } else {
                    let mut max_pos = bisect::left(&maxes, &m)?;

                    let max_idx = if max_pos == maxes.len() {
                        max_pos -= 1;
                        lists[max_pos].len()
                    } else {
                        bisect::left(&lists[max_pos], &m)?
                    };
                    Ok(Pos::new(max_pos, max_idx))
                }
            })
            .unwrap_or_else(|| {
                let max_pos = maxes.len() - 1;
                let max_idx = lists[max_pos].len();
                Ok(Pos::new(max_pos, max_idx))
            })?;

        if bounds.min.pos > bounds.max.pos
            || (bounds.min.pos == bounds.max.pos && bounds.min.idx >= bounds.max.idx)
        {
            Ok(None)
        } else {
            Ok(Some(bounds))
        }
    }

    pub(super) fn get_islice_specs(
        data: &mut ListsData,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<Bounds>> {
        let length = data.len as isize;
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
                    bounds.max.set_from_pos(indices.stop, data)?
                };

                Ok(Some(bounds))
            }
        }
    }
}
