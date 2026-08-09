use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PySlice, PySliceIndices},
};
use std::cmp::Ordering;
use tap::prelude::*;

use crate::collections::sorted::{bisect, errors, iter::IsliceBounds};
pub(super) struct ListsData {
    pub(super) lists: Vec<Vec<Py<PyAny>>>,
    pub(super) maxes: Vec<Py<PyAny>>,
    pub(super) idx: Vec<usize>,
    pub(super) len: usize,
    pub(super) offset: usize,
}
impl ListsData {
    pub fn new() -> Self {
        Self {
            lists: Vec::new(),
            maxes: Vec::new(),
            idx: Vec::new(),
            len: 0,
            offset: 0,
        }
    }
    #[inline]
    pub fn collapse(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.lists
            .iter()
            .flatten()
            .map(|x| x.clone_ref(py))
            .collect()
    }
    #[inline]
    pub fn concat(&self, py: Python<'_>, other: Bound<'_, PyAny>) -> PyResult<Vec<Py<PyAny>>> {
        let mut values = self.collapse(py);
        let mut new_vals = other
            .try_iter()?
            .map(|x| x?.unbind().clone_ref(py).pipe(Ok))
            .collect::<PyResult<Vec<_>>>()?;
        values.append(new_vals.as_mut());
        Ok(values)
    }
    #[inline]
    pub fn repeat(&self, py: Python<'_>, num: usize) -> Vec<Py<PyAny>> {
        let values = self.collapse(py);
        (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect::<Vec<_>>()
    }

    #[inline]
    pub fn clear(&mut self) {
        self.lists.clear();
        self.maxes.clear();
        self.idx.clear();
        self.len = 0;
        self.offset = 0;
    }

    pub(crate) fn getitem_from_int<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let len_last = self
            .lists
            .last()
            .ok_or(PyIndexError::new_err("list index out of range"))?
            .len() as isize;
        match (index, self.len != 0) {
            (0, true) => self.lists[0][0].clone_ref(py).into_bound(py).pipe(Ok),
            (-1, true) => self
                .lists
                .last()
                .unwrap()
                .last()
                .unwrap()
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, false) => {
                let msg = "list index out of range";
                Err(PyIndexError::new_err(msg))
            }
            (_, true) if 0 <= index && index < self.lists[0].len() as isize => self.lists[0]
                [index as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, true) if -len_last < index && index < 0 => self.lists.last().unwrap()
                [(len_last + index) as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            _ => {
                let (pos, idx) = self.pos(index)?;
                self.lists[pos][idx as usize]
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    pub(crate) fn getitem_from_slice<'py>(
        &mut self,
        py: Python<'py>,
        slice: Bound<'py, PySlice>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.len as isize)?;
        let stop_eq_len = stop == self.len as isize;
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => self.collapse(py).pipe(Ok),
            (1, Ordering::Less) => {
                let (start_pos, start_idx) = self.pos(start)?;
                let start_list = &self.lists[start_pos];
                let stop_idx = start_idx + stop - start;
                match (start_list.len() as isize >= stop_idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (true, _) => start_list[start_idx as usize..stop_idx as usize]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>()
                        .pipe(Ok),
                    (false, true) => {
                        let stop_pos = self.lists.len() - 1;
                        let stop_idx = (&self.lists)[stop_pos].len();
                        get_slice(&self, py, start_pos, stop_pos, start_idx as usize, stop_idx)
                    }
                    (false, false) => {
                        let (stop_pos, stop_idx) = self.pos(stop)?;
                        get_slice(
                            &self,
                            py,
                            start_pos,
                            stop_pos,
                            start_idx as usize,
                            stop_idx as usize,
                        )
                    }
                }
            }
            (-1, Ordering::Greater) => {
                let mut result =
                    self.getitem_from_slice(py, PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .map(|i| self.getitem_from_int(py, i).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>(),
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (_, Ordering::Less | Ordering::Equal) => Ok(Vec::new()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.getitem_from_int(py, i).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            }
        }
    }

    /// Convert an index into an index pair (lists index, sublist index).

    /// This pair can be used to access the corresponding lists position.

    /// Many queries require the index be built. Details of the index are
    /// described in ``SortedList._build_index``.

    /// Indexing requires traversing the tree to a leaf node. Each node has two
    /// children which are easily computable. Given an index, pos, the
    /// left-child is at ``pos * 2 + 1`` and the right-child is at ``pos * 2 +
    /// 2``.

    /// When the index is less than the left-child, traversal moves to the
    /// left sub-tree. Otherwise, the index is decremented by the left-child
    /// and traversal moves to the right sub-tree.

    /// At a child node, the indexing pair is computed from the relative
    /// position of the child node as compared with the offset and the remaining
    /// index.

    /// For example, using the index from ``SortedList._build_index``::

    ///            _index = 14 5 9 3 2 4 5
    ///            _offset = 3

    /// Tree::

    ///                 14
    ///              5      9
    ///            3   2  4   5

    /// Indexing position 8 involves iterating like so:

    /// 1. Starting at the root, position 0, 8 is compared with the left-child
    ///    node (5) which it is greater than. When greater the index is
    ///    decremented and the position is updated to the right child node.

    /// 2. At node 9 with index 3, we again compare the index to the left-child
    ///    node with value 4. Because the index is the less than the left-child
    ///    node, we simply traverse to the left.

    /// 3. At node 4 with index 3, we recognize that we are at a leaf node and
    ///    stop iterating.

    /// 4. To compute the sublist index, we subtract the offset from the index
    ///    of the leaf node: 5 - 3 = 2. To compute the index in the sublist, we
    ///    simply use the index remaining from iteration. In this case, 3.

    /// The final index pair from our example is (2, 3) which corresponds to
    /// index 8 in the sorted list.
    pub(crate) fn pos(&mut self, mut idx: isize) -> PyResult<(usize, isize)> {
        if idx < 0 {
            if (-idx) <= self.lists.last().unwrap().len() as isize {
                return Ok((
                    self.lists.len() - 1,
                    self.lists.last().unwrap().len() as isize + idx,
                ));
            }

            idx += self.len as isize;

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= self.len as isize {
            return errors::out_of_range_err();
        }

        if idx < self.lists[0].len() as isize {
            return Ok((0, idx));
        }

        if self.idx.is_empty() {
            self.build_index()?;
        }
        let pos = self.idx.pipe_ref_mut(|index| {
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

        Ok((pos - self.offset, idx))
    }

    /// Build a positional index for indexing the sorted list.
    /// Indexes are represented as binary trees in a dense array notation similar to a binary heap.

    /// For example, given a lists representation storing integers:

    ///     0: [1, 2, 3]
    ///     1: [4, 5]
    ///     2: [6, 7, 8, 9]
    ///     3: [10, 11, 12, 13, 14]

    /// The first transformation maps the sub-lists by their length.\
    /// The first row of the index is the length of the sub-lists::

    ///     0: [3, 2, 4, 5]

    /// Each row after that is the sum of consecutive pairs of the previous row:

    ///     1: [5, 9]
    ///     2: [14]

    /// Finally, the index is built by concatenating these lists together:

    ///     _index = [14, 5, 9, 3, 2, 4, 5]

    /// An offset storing the start of the first row is also stored:

    ///     _offset = 3
    /// When built, the index can be used for efficient indexing into the list.
    pub(crate) fn build_index(&mut self) -> PyResult<()> {
        let row0 = self.lists.iter().map(|x| x.len()).collect::<Vec<usize>>();

        if row0.len() == 1 {
            self.idx.extend(row0);
            self.offset = 0;
            return Ok(());
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            self.idx.clear();
            self.idx.extend(combined);
            self.offset = 1;
            Ok(())
        } else {
            let size = 1usize << ((row1.len() - 1).ilog2() + 1);
            row1.resize(size, 0);

            let mut tree = vec![row0, row1];
            while tree.last().unwrap().len() > 1 {
                let row = tree
                    .last()
                    .unwrap()
                    .chunks(2)
                    .map(|pair| pair.iter().sum())
                    .collect();
                tree.push(row);
            }

            let flat = tree.into_iter().rev().flatten();
            self.idx.extend(flat);
            self.offset = size * 2 - 1;
            Ok(())
        }
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
    pub(crate) fn loc(&mut self, mut pos: usize, idx: isize) -> PyResult<isize> {
        if pos == 0 {
            Ok(idx)
        } else {
            if self.idx.is_empty() {
                self.build_index()?;
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            pos += self.offset;
            // Iterate until reaching the root of the index tree at pos = 0.
            let total = self.idx.pipe_ref_mut(|idx| {
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

            Ok(total + idx)
        }
    }
}

fn get_slice(
    data: &ListsData,
    py: Python<'_>,
    start_pos: usize,
    stop_pos: usize,
    start_idx: usize,
    stop_idx: usize,
) -> PyResult<Vec<Py<PyAny>>> {
    let new_items = data.lists[start_pos + 1..stop_pos]
        .iter()
        .flatten()
        .map(|x| x.clone_ref(py));
    data.lists[start_pos][start_idx..]
        .iter()
        .map(|x| x.clone_ref(py))
        .chain(new_items)
        .chain(
            data.lists[stop_pos][0..stop_idx]
                .iter()
                .map(|x| x.clone_ref(py)),
        )
        .collect::<Vec<_>>()
        .pipe(Ok)
}

pub(super) fn get_irange_specs(
    lists: &Vec<Vec<Py<PyAny>>>,
    maxes: &Vec<Py<PyAny>>,
    minimum: Option<Bound<'_, PyAny>>,
    maximum: Option<Bound<'_, PyAny>>,
    inclusive: (bool, bool),
) -> PyResult<Option<IsliceBounds>> {
    if maxes.is_empty() {
        return Ok(None);
    }

    // Calculate the minimum (pos, idx) pair. By default this location
    // will be inclusive in our calculation.
    let (min_pos, min_idx) = match minimum {
        None => (0, 0),
        Some(minimum) => {
            if inclusive.0 {
                let min_pos = bisect::left(&maxes, &minimum)?;

                if min_pos == maxes.len() {
                    return Ok(None);
                }

                let min_idx = bisect::left(&lists[min_pos], &minimum)?;
                (min_pos, min_idx)
            } else {
                let min_pos = bisect::right(&maxes, &minimum)?;

                if min_pos == maxes.len() {
                    return Ok(None);
                }

                let min_idx = bisect::right(&lists[min_pos], &minimum)?;
                (min_pos, min_idx)
            }
        }
    };

    // Calculate the maximum (pos, idx) pair. By default this location
    // will be exclusive in our calculation.
    let (max_pos, max_idx) = maximum
        .map(|m| {
            if inclusive.1 {
                let mut max_pos = bisect::right(&maxes, &m)?;

                let max_idx = if max_pos == maxes.len() {
                    max_pos -= 1;
                    lists[max_pos].len()
                } else {
                    bisect::right(&lists[max_pos], &m)?
                };
                Ok::<_, PyErr>((max_pos, max_idx))
            } else {
                let mut max_pos = bisect::left(&maxes, &m)?;

                let max_idx = if max_pos == maxes.len() {
                    max_pos -= 1;
                    lists[max_pos].len()
                } else {
                    bisect::left(&lists[max_pos], &m)?
                };
                Ok((max_pos, max_idx))
            }
        })
        .unwrap_or_else(|| {
            let max_pos = maxes.len() - 1;
            let max_idx = lists[max_pos].len();
            Ok((max_pos, max_idx))
        })?;
    IsliceBounds::from_irange_spec(min_pos, min_idx, max_pos, max_idx).pipe(Ok)
}
