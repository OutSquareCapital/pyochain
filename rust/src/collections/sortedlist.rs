use crate::pyo3_ext::{prelude::*, pylibs};
use crate::seq::{IntoPyochain, PyoVec};
use either::Either;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::types::{PySequence, PySlice, PySliceIndices};
use pyo3::{prelude::*, types::PyList};
use pyochain_macros::py_abc;
use std::cmp::Ordering;
use tap::Pipe;
const DEFAULT_LOAD_FACTOR: usize = 1000;
trait InnerSortedRs {
    fn get_lists(&self, py: Python<'_>) -> Py<PyoVec>;
    fn get_idx(&self, py: Python<'_>) -> Py<PyoVec>;
    fn get_maxes(&self, py: Python<'_>) -> Py<PyoVec>;
    fn get_offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
    fn get_len(&self) -> usize;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl InnerSortedRs for $t {
            fn get_lists(&self, py: Python<'_>) -> Py<PyoVec> {
                self.lists.clone_ref(py)
            }
            fn get_idx(&self, py: Python<'_>) -> Py<PyoVec> {
                self.idx.clone_ref(py)
            }
            fn get_offset(&self) -> usize {
                self.offset
            }
            fn set_offset(&mut self, offset: usize) {
                self.offset = offset;
            }
            fn get_len(&self) -> usize {
                self.len
            }
            fn get_maxes(&self, py: Python<'_>) -> Py<PyoVec> {
                self.maxes.clone_ref(py)
            }
        }
    };
}
impl_inner_sorted_rs!(InnerLists);
impl_inner_sorted_rs!(InnerKeyLists);

#[py_abc(InnerLists, InnerKeyLists)]
trait InnerSorted: Sized + InnerSortedRs {
    fn bisect_left(&mut self, value: Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&mut self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    fn clear(&mut self, py: Python<'_>) -> ();
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool>;
    fn delete(&mut self, py: Python<'_>, pos: usize, idx: usize) -> PyResult<()>;
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()>;
    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&mut self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn count(&mut self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    fn update(&mut self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &mut self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
    fn collapse_lists<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyoVec>> {
        let init = PyList::empty(py).into_sequence();
        self.get_lists(py)
            .get()
            .inner
            .bind(py)
            .iter()
            .try_fold(init, |acc, x| {
                unsafe { x.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .as_sequence()
                    .pipe(|x| acc.in_place_concat(x))?;
                Ok::<_, PyErr>(acc)
            })
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?
            .into_pyochain()
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
    fn build_index(&mut self, py: Python<'_>) -> PyResult<()> {
        let idx = self.get_idx(py).get().inner.clone_ref(py).into_bound(py);
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);

        let row0 = lists
            .iter()
            .map(|x| x.len())
            .collect::<PyResult<Vec<usize>>>()?;

        if row0.len() == 1 {
            idx.set_slice(0, idx.len(), PyList::new(py, row0)?.as_any())?;
            self.set_offset(0);
            return Ok(());
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1
                .into_iter()
                .chain(row0)
                .try_fold(PyList::empty(py), iterator_into_list)?;
            idx.set_slice(0, idx.len(), combined.as_any())?;
            self.set_offset(1);
            return Ok(());
        }

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

        let flat = tree
            .into_iter()
            .rev()
            .flatten()
            .try_fold(PyList::empty(py), iterator_into_list)?
            .into_any();
        idx.set_slice(0, idx.len(), &flat)?;
        self.set_offset(size * 2 - 1);
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
    fn loc(&mut self, py: Python<'_>, mut pos: usize, idx: isize) -> PyResult<isize> {
        if pos == 0 {
            Ok(idx)
        } else {
            let index = self.get_idx(py).get().inner.clone_ref(py).into_bound(py);

            if index.is_empty() {
                self.build_index(py)?;
            }
            let mut total = 0;
            // Increment pos to point in the index to len(self.lists[pos]).
            pos += self.get_offset();
            // Iterate until reaching the root of the index tree at pos = 0.
            while pos != 0 {
                // Right-child nodes are at even indices. At such indices
                // account the total below the left child node.

                if pos % 2 == 0 {
                    total += index.get_item(pos - 1)?.extract::<isize>()?;
                }

                // Advance pos to the parent node.

                pos = (pos - 1) >> 1;
            }

            Ok(total + idx)
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
    fn pos(&mut self, py: Python<'_>, mut idx: isize) -> PyResult<(usize, isize)> {
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);
        if idx < 0 {
            if (-idx) <= lists.last()?.len()? as isize {
                return Ok((lists.len() - 1, lists.last()?.len()? as isize + idx));
            }

            idx += self.get_len() as isize;

            if idx < 0 {
                return out_of_range_err();
            }
        } else if idx >= self.get_len() as isize {
            return out_of_range_err();
        }

        if idx < lists.get_item(0)?.len()? as isize {
            return Ok((0, idx));
        }

        let index = self.get_idx(py).get().inner.clone_ref(py).into_bound(py);

        if index.is_empty() {
            self.build_index(py)?;
        }

        let mut pos = 0;
        let mut child = 1;
        let len_index = index.len();

        while child < len_index {
            let index_child = index.get_item(child)?.extract::<isize>()?;

            if idx < index_child {
                pos = child;
            } else {
                idx -= index_child;
                pos = child + 1;
            }

            child = (pos << 1) + 1
        }

        return Ok((pos - self.get_offset(), idx));
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        if self.get_len() == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        } else {
            let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);

            let len_last = lists.last()?.len()? as isize;
            let val = match index {
                0 => {
                    let val = lists.get_item(0)?.get_item(0)?;
                    self.delete(py, 0, 0)?;
                    val
                }

                -1 => {
                    let pos = lists.len() - 1;
                    let loc = lists.get_item(pos)?.len()? - 1;
                    let val = lists.get_item(pos)?.get_item(loc)?;
                    self.delete(py, pos, loc)?;
                    val
                }

                _ if 0 <= index && index < lists.get_item(0)?.len()? as isize => {
                    let val = lists.get_item(0)?.get_item(index)?;
                    self.delete(py, 0, index as usize)?;
                    val
                }
                _ if -len_last < index && index < 0 => {
                    let pos = lists.len() - 1;
                    let loc = len_last + index;
                    let val = lists.get_item(pos)?.get_item(loc)?;
                    self.delete(py, pos, loc as usize)?;
                    val
                }
                _ => {
                    let (pos, idx) = self.pos(py, index)?;
                    let val = lists.get_item(pos)?.get_item(idx)?;
                    self.delete(py, pos, idx as usize)?;
                    val
                }
            };
            Ok(val)
        }
    }

    fn getitem<'py>(
        &mut self,
        py: Python<'py>,
        index: Either<isize, Bound<'py, PySlice>>,
    ) -> PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>> {
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);

        match index {
            Either::Right(slice) => self
                .getitem_from_slice(py, &lists, slice)
                .map(Either::Right),
            Either::Left(index) => self.getitem_from_int(py, &lists, index).map(Either::Left),
        }
    }
    fn getitem_from_int<'py>(
        &mut self,
        py: Python<'py>,
        lists: &Bound<'py, PyList>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slf_len = self.get_len();
        let len_last = lists.last()?.len()? as isize;
        match (index, slf_len != 0) {
            (0, true) => {
                return lists.get_item(0)?.get_item(0);
            }
            (-1, true) => {
                return lists.last()?.get_item(-1);
            }
            (_, false) => {
                let msg = "list index out of range";
                Err(PyIndexError::new_err(msg))
            }
            (_, true) if 0 <= index && index < lists.get_item(0)?.len()? as isize => {
                return lists.get_item(0)?.get_item(index);
            }
            (_, true) if -len_last < index && index < 0 => {
                return lists.last()?.get_item(len_last + index);
            }
            _ => {
                let (pos, idx) = self.pos(py, index)?;
                return lists.get_item(pos)?.get_item(idx);
            }
        }
    }
    fn getitem_from_slice<'py>(
        &mut self,
        py: Python<'py>,
        lists: &Bound<'py, PyList>,
        slice: Bound<'py, PySlice>,
    ) -> PyResult<Bound<'py, PyoVec>> {
        let slice_result =
            |start_pos: usize, stop_pos: usize, start_idx: usize, stop_idx: usize| {
                let prefix = lists
                    .get_item(start_pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py)
                    .get_slice(start_idx, usize::MAX)
                    .into_sequence();
                lists
                    .get_slice(start_pos + 1, stop_pos)
                    .into_iter()
                    .map(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
                    .try_fold(prefix, |acc, item| {
                        acc.in_place_concat(&item)?;
                        Ok::<_, PyErr>(acc)
                    })?
                    .iadd(
                        lists
                            .get_item(stop_pos)
                            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                            .get()
                            .inner
                            .bind(py)
                            .get_slice(0, stop_idx)
                            .into_pyochain()?,
                    )
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?
                    .into_pyochain()
            };

        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.get_len() as isize)?;
        let stop_eq_len = stop == self.get_len() as isize;
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => self.collapse_lists(py),
            (1, Ordering::Less) => {
                let (start_pos, start_idx) = self.pos(py, start)?;
                let start_list = lists
                    .get_item(start_pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py);
                let stop_idx = start_idx + stop - start;
                match (start_list.len() as isize >= stop_idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (true, _) => start_list
                        .get_slice(start_idx as usize, stop_idx as usize)
                        .into_pyochain(),
                    (false, true) => {
                        let stop_pos = lists.len() - 1;
                        let stop_idx = lists.get_item(stop_pos)?.len()?;
                        slice_result(start_pos, stop_pos, start_idx as usize, stop_idx)
                    }
                    (false, false) => {
                        let (stop_pos, stop_idx) = self.pos(py, stop)?;
                        slice_result(start_pos, stop_pos, start_idx as usize, stop_idx as usize)
                    }
                }
            }
            (-1, Ordering::Greater) => {
                let result =
                    self.getitem_from_slice(py, lists, PySlice::new(py, stop + 1, start + 1, 1))?;
                result.get().inner.bind(py).reverse()?;
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .map(|i| self.getitem_from_int(py, lists, i))
                .try_fold(PyList::empty(py), try_iterator_into_list)?
                .into_pyochain(),
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (_, Ordering::Less | Ordering::Equal) => PyList::empty(py).into_pyochain(),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.getitem_from_int(py, lists, i))
                    .try_fold(PyList::empty(py), try_iterator_into_list)?
                    .into_pyochain()
            }
        }
    }

    fn delitem(
        &mut self,
        py: Python<'_>,
        index: Either<isize, Bound<'_, PySlice>>,
    ) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.delitem_from_slice(py, slice),
            Either::Left(index) => {
                let (pos, idx) = self.pos(py, index)?;
                self.delete(py, pos, idx as usize)
            }
        }
    }
    fn delitem_from_slice(&mut self, py: Python<'_>, slice: Bound<'_, PySlice>) -> PyResult<()> {
        let length = self.get_len() as isize;
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(length)?;
        match (step, start.cmp(&stop)) {
            (1, Ordering::Less) if start == 0 && stop == length => {
                self.clear(py);
                Ok(())
            }
            (1, Ordering::Less) if length <= 8 * (stop - start) => {
                let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);
                let values = self.getitem_from_slice(py, &lists, PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice =
                        self.getitem_from_slice(py, &lists, PySlice::new(py, stop, length, 1))?;
                    values.iadd(new_slice)?;
                }
                self.clear(py);
                self.update(values.as_any())?;
                Ok(())
            }
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .rev()
                .try_for_each(|idx| {
                    let (pos, idx) = self.pos(py, idx)?;
                    self.delete(py, pos, idx as usize)
                }),
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        let (pos, idx) = self.pos(py, idx)?;
                        self.delete(py, pos, idx as usize)
                    })
            }
        }
    }
}

#[pyclass(generic)]
pub struct InnerLists {
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerLists {
    #[new]
    fn new(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }
}
impl InnerSorted for InnerLists {
    fn clear(&mut self, py: Python<'_>) -> () {
        self.len = 0;
        self.lists.get().clear(py);
        self.maxes.get().clear(py);
        self.idx.get().clear(py);
        self.offset = 0;
    }

    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let maxes = self.maxes.bind(value.py());

        if maxes.is_empty()? {
            return Ok(false);
        }

        let pos = bisect::bisect_left(maxes, &value)?;

        if maxes.len()?.eq(&pos) {
            return Ok(false);
        }

        let lists = self.lists.bind(value.py());
        let v = &lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;

        lists.get_item(pos)?.get_item(idx)?.eq(value)
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let load = self.load;
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);
        let index = self.idx.get().inner.bind(py);

        if lists.get_item(pos)?.len()?.gt(&(load << 1)) {
            let maxes = self.maxes.get().inner.bind(py);

            let lists_pos = lists
                .get_item(pos)?
                .pipe(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let half = lists_pos.get_slice(load, usize::MAX);
            lists_pos.del_slice(load, usize::MAX)?;
            maxes.set_item(pos, lists_pos.last()?)?;
            let last = half.last()?;
            lists.insert(pos + 1, &half.into_pyochain()?)?;
            maxes.insert(pos + 1, last)?;

            index.clear();
            Ok(())
        } else if !index.is_empty() {
            let mut child = self.offset + pos;
            while child != 0 {
                index.set_item(child, index.get_item(child)?.iadd(1)?)?;
                child = (child - 1) >> 1;
            }
            index.set_item(0, index.get_item(0)?.iadd(1)?)?;
            Ok(())
        } else {
            Ok(())
        }
    }

    fn delete(&mut self, py: Python<'_>, mut pos: usize, idx: usize) -> PyResult<()> {
        let lists = self.lists.bind(py).get().inner.bind(py);
        let maxes = self.maxes.bind(py);
        let index = self.idx.bind(py).get().inner.bind(py);

        let lists_pos = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);

        lists_pos.del_item(idx)?;
        self.len -= 1;

        let len_lists_pos = lists_pos.len();

        if len_lists_pos > (self.load >> 1) {
            maxes.set_item(pos, lists_pos.last()?)?;

            if !index.is_empty() {
                let mut child = self.offset + pos;
                while child > 0 {
                    index.set_item(child, index.get_item(child)?.isub(1)?)?;
                    child = (child - 1) >> 1
                }
                index.set_item(0, index.get_item(0)?.isub(1)?)?;
            }
            Ok(())
        } else if lists.len() > 1 {
            if pos == 0 {
                pos += 1;
            }

            let prev = (pos - 1) as usize;
            lists
                .get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(lists.get_item(pos)?)?;
            maxes.set_item(prev, lists.get_item(prev)?.get_item(-1)?)?;

            lists.del_item(pos)?;
            maxes.del_item(pos)?;
            index.clear();

            self.expand(py, prev)
        } else if len_lists_pos != 0 {
            maxes.set_item(pos, lists_pos.last()?)?;
            Ok(())
        } else {
            lists.del_item(pos)?;
            maxes.del_item(&pos)?;
            index.clear();
            Ok(())
        }
    }
    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let lists = self.lists.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        if !maxes.is_empty() {
            let mut pos = bisect::bisect_right(self.maxes.bind(py), &value)?;

            if pos == maxes.len() {
                pos -= 1;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&value)?;
                maxes.set_item(pos, &value)?;
            } else {
                let vector = &lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

                let res = bisect::bisect_right(&vector, &value)?;
                vector.get().insert(res, &value)?;
            }

            self.expand(py, pos)?;
        } else {
            lists.append(PyList::new(py, [&value])?.into_pyochain()?)?;
            maxes.append(&value)?;
        }

        self.len += 1;
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(());
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(());
        }

        let lists = self.lists.get().inner.bind(value.py());
        let v = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;

        if lists.get_item(pos)?.get_item(idx)?.eq(value)? {
            self.delete(py, pos, idx)
        } else {
            Ok(())
        }
    }

    fn remove(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            not_in_list_err(value)
        } else {
            let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

            if pos == maxes.len() {
                not_in_list_err(value)
            } else {
                let lists = self.lists.get().inner.bind(py);
                let v = &lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

                let idx = bisect::bisect_left(&v, &value)?;

                if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                    self.delete(py, pos, idx)
                } else {
                    not_in_list_err(value)
                }
            }
        }
    }

    fn bisect_left(&mut self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(self.len as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_left(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn bisect_right(&mut self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_right(self.maxes.bind(py), &value)?;

        if pos == maxes.len() {
            return Ok(self.len as isize);
        }
        let v = self
            .lists
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

        let idx = bisect::bisect_right(&v, &value)?;
        self.loc(py, pos, idx as isize)
    }

    fn count(&mut self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos_left = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos_left == maxes.len() {
            return Ok(0);
        }

        let lists = self.lists.get().inner.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_left = bisect::bisect_left(&v_left, &value)?;
        let pos_right = bisect::bisect_right(self.maxes.bind(py), &value)?;

        if pos_right == maxes.len() {
            return Ok(self.len - self.loc(py, pos_left, idx_left as isize)? as usize);
        }
        let v_right = lists
            .get_item(pos_right)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_right = bisect::bisect_right(&v_right, &value)?;

        if pos_left == pos_right {
            return Ok(idx_right - idx_left);
        }

        let right = self.loc(py, pos_right, idx_right as isize)?;
        let left = self.loc(py, pos_left, idx_left as isize)?;
        Ok((right - left) as usize)
    }

    fn index(
        &mut self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let len_ = self.len as isize;

        if len_ == 0 {
            return is_not_in_list_err(value);
        }

        let mut start = start.unwrap_or(0);
        if start < 0 {
            start += len_;
        }
        start = start.max(0);

        let mut stop = stop.unwrap_or(len_);
        if stop < 0 {
            stop += len_;
        }
        stop = stop.min(len_);

        if stop <= start {
            return is_not_in_list_err(value);
        }

        let maxes = self.maxes.get().inner.bind(py);
        let pos_left = bisect::bisect_left(self.maxes.bind(py), &value)?;

        if pos_left == maxes.len() {
            return is_not_in_list_err(value);
        }

        let lists = self.lists.get().inner.bind(py);
        let v_left = lists
            .get_item(pos_left)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx_left = bisect::bisect_left(&v_left, &value)?;

        if lists.get_item(pos_left)?.get_item(idx_left)?.ne(&value)? {
            return is_not_in_list_err(value);
        }

        stop -= 1;
        let left = self.loc(py, pos_left, idx_left as isize)?;

        if start <= left {
            if left <= stop {
                return Ok(left);
            }
        } else {
            let right = self.bisect_right(&value)? - 1;

            if start <= right {
                return Ok(start);
            }
        }

        is_not_in_list_err(value)
    }

    fn update(&mut self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let maxes = self.get_maxes(py).get().inner.clone_ref(py).into_bound(py);
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|iterator| pylibs::builtins::sorted(&iterator, false))?;

        if !maxes.is_empty() {
            if values.len() * 4 >= self.len {
                lists.append(values.into_pyochain()?)?;
                values = self
                    .collapse_lists(py)?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py);
                values.sort()?;
                self.clear(py);
            } else {
                for val in values {
                    self.add(val)?;
                }
                return Ok(());
            }
        }

        let load = self.load;
        let values_len = values.len();

        (0..values_len)
            .step_by(load)
            .map(|pos| values.get_slice(pos, pos + load).into_pyochain())
            .try_fold(lists, try_iterator_into_list)?
            .iter()
            .map(|x| {
                unsafe { x.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .last()
            })
            .try_fold(maxes, try_iterator_into_list)?;
        self.len = values_len;
        self.idx.get().clear(py);
        Ok(())
    }
}

#[pyclass(generic)]
pub struct InnerKeyLists {
    #[pyo3(get, set)]
    key: Py<PyAny>,
    #[pyo3(get, set)]
    keys: Py<PyoVec>,
    #[pyo3(get, set)]
    lists: Py<PyoVec>,
    #[pyo3(get, set)]
    maxes: Py<PyoVec>,
    #[pyo3(get, set)]
    idx: Py<PyoVec>,
    #[pyo3(get, set)]
    len: usize,
    #[pyo3(get, set)]
    load: usize,
    #[pyo3(get, set)]
    offset: usize,
}
#[pymethods]
impl InnerKeyLists {
    #[new]
    fn new(key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = key.py();
        Ok(Self {
            key: key.unbind(),
            keys: PyoVec::new_bound(py)?.unbind(),
            lists: PyoVec::new_bound(py)?.unbind(),
            maxes: PyoVec::new_bound(py)?.unbind(),
            idx: PyoVec::new_bound(py)?.unbind(),
            len: 0,
            load: DEFAULT_LOAD_FACTOR,
            offset: 0,
        })
    }
}
impl InnerSorted for InnerKeyLists {
    fn clear(&mut self, py: Python<'_>) -> () {
        self.len = 0;
        self.lists.get().clear(py);
        self.keys.get().clear(py);
        self.maxes.get().clear(py);
        self.idx.get().clear(py);
    }
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let maxes = self.maxes.bind(py);

        if maxes.is_empty()? {
            return Ok(false);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(maxes, &key)?;

        if pos == maxes.len()? {
            return Ok(false);
        }

        let lists = self.lists.bind(py);
        let keys = self.keys.bind(py);
        let v = &keys
            .get_item(&pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v, &key)?;

        let len_keys = keys.len()?;
        let mut len_sublist = keys.get_item(&pos)?.len()?;

        loop {
            if keys.get_item(&pos)?.get_item(&idx)?.ne(&key)? {
                return Ok(false);
            }
            if lists.get_item(&pos)?.get_item(&idx)?.eq(&value)? {
                return Ok(true);
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(false);
                }
                len_sublist = keys.get_item(&pos)?.len()?;
                idx = 0;
            }
        }
    }
    fn delete(&mut self, py: Python<'_>, mut pos: usize, idx: usize) -> PyResult<()> {
        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        let index = self.idx.get().inner.bind(py);
        let keys_pos = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);
        let lists_pos = lists
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py);

        keys_pos.del_item(idx)?;
        lists_pos.del_item(idx)?;
        self.len -= 1;

        let len_keys_pos = keys_pos.len();

        if len_keys_pos > (self.load >> 1) {
            maxes.set_item(pos, keys_pos.last()?)?;

            if !index.is_empty() {
                let mut child = self.offset + pos;
                while child > 0 {
                    index.set_item(child, index.get_item(child)?.isub(1)?)?;
                    child = (child - 1) >> 1;
                }
                index.set_item(0, index.get_item(0)?.isub(1)?)?;
            }
            Ok(())
        } else if keys.len() > 1 {
            if pos == 0 {
                pos += 1
            }

            let prev = pos - 1;
            keys.get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(keys.get_item(pos)?)?;
            lists
                .get_item(prev)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .extend(lists.get_item(pos)?)?;
            maxes.set_item(prev, keys.get_item(prev)?.get_item(-1)?)?;

            lists.del_item(pos)?;
            keys.del_item(pos)?;
            maxes.del_item(pos)?;
            index.clear();

            self.expand(py, prev)
        } else if len_keys_pos != 0 {
            maxes.set_item(pos, keys_pos.last()?)
        } else {
            lists.del_item(pos)?;
            keys.del_item(pos)?;
            maxes.del_item(pos)?;
            index.clear();
            Ok(())
        }
    }
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()> {
        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);
        let index = self.idx.get().inner.clone_ref(py).into_bound(py);

        if keys.get_item(pos)?.len()? > self.load << 1 {
            let maxes = self.maxes.get().inner.bind(py);
            let load = self.load;

            let lists_pos = lists
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let keys_pos = keys
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                .get()
                .inner
                .clone_ref(py)
                .into_bound(py);
            let half = lists_pos.get_slice(load, lists_pos.len()).into_pyochain()?;
            let half_keys = keys_pos.get_slice(load, keys_pos.len()).into_pyochain()?;
            lists_pos.del_slice(load, usize::MAX)?;
            keys_pos.del_slice(load, usize::MAX)?;
            maxes.set_item(pos, keys_pos.last()?)?;

            lists.insert(pos + 1, half)?;
            keys.insert(pos + 1, &half_keys)?;
            maxes.insert(pos + 1, half_keys.get_item(half_keys.len()? - 1)?)?;
            index.clear();
            Ok(())
        } else if !index.is_empty() {
            let mut child = self.offset + pos;
            while child != 0 {
                index.set_item(child, index.get_item(child)?.iadd(1)?)?;
                child = (child - 1) >> 1;
            }
            index.set_item(0, index.get_item(0)?.iadd(1)?)?;
            Ok(())
        } else {
            Ok(())
        }
    }
    fn add(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let key = self.key.bind(py).call1((&value,))?;
        let lists = self.lists.get().inner.bind(py);
        let maxes = self.maxes.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);

        if !maxes.is_empty() {
            let mut pos = bisect::bisect_right(self.maxes.bind(py), &key)?;

            if pos == maxes.len() {
                pos -= 1;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&value)?;
                keys.get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .append(&key)?;
                maxes.set_item(pos, &key)?;
            } else {
                let v = &keys
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
                let idx = bisect::bisect_right(&v, &key)?;
                lists
                    .get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .insert(idx, &value)?;
                keys.get_item(pos)
                    .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?
                    .get()
                    .insert(idx, &key)?;
            }

            self.expand(py, pos)?;
        } else {
            lists.append(PyList::new(py, [value])?.into_pyochain()?)?;
            keys.append(PyList::new(py, [&key])?.into_pyochain()?)?;
            maxes.append(key)?;
        }

        self.len += 1;
        Ok(())
    }

    fn discard(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(());
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(());
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v = &keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                break;
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                self.delete(py, pos, idx)?;
                break;
            } else {
                idx += 1;
                if idx == len_sublist {
                    pos += 1;
                    if pos == len_keys {
                        break;
                    } else {
                        len_sublist = keys.get_item(pos)?.len()?;
                        idx = 0;
                        continue;
                    }
                }
            }
        }
        Ok(())
    }

    fn remove(&mut self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return not_in_list_err(value);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return not_in_list_err(value);
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v = &keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;

        let mut idx = bisect::bisect_left(&v, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return not_in_list_err(value);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                self.delete(py, pos, idx)?;
                break;
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return not_in_list_err(value);
                }
                len_sublist = keys.get_item(pos)?.len()?;
                idx = 0
            }
        }
        Ok(())
    }
    fn bisect_left(&mut self, value: Bound<'_, PyAny>) -> PyResult<isize> {
        self.key
            .bind(value.py())
            .call1((value,))
            .and_then(|x| self.bisect_key_left(x))
    }

    fn bisect_right(&mut self, value: &Bound<'_, PyAny>) -> PyResult<isize> {
        self.key
            .bind(value.py())
            .call1((value,))
            .and_then(|x| self.bisect_key_right(x))
    }

    fn count(&mut self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        let py = value.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(0);
        }

        let lists = self.lists.get().inner.bind(py);
        let keys = self.keys.get().inner.bind(py);
        let v_left = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v_left, &key)?;
        let mut total = 0;
        let len_keys = keys.len();
        let mut len_sublist = keys.get_item(pos)?.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return Ok(total);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                total += 1;
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return Ok(total);
                }
                len_sublist = keys.get_item(pos)?.len()?;
                idx = 0;
            }
        }
    }
    fn index(
        &mut self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = value.py();
        let len_ = self.len as isize;

        if len_ == 0 {
            return is_not_in_list_err(value);
        }

        let mut start = start.unwrap_or(0);
        if start < 0 {
            start += len_;
        }
        start = start.max(0);

        let mut stop = stop.unwrap_or(len_);
        if stop < 0 {
            stop += len_;
        }
        stop = stop.min(len_);

        if stop <= start {
            return is_not_in_list_err(value);
        }

        let maxes = self.maxes.get().inner.bind(py);
        let key = self.key.bind(py).call1((&value,))?;
        let mut pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return is_not_in_list_err(value);
        }

        stop -= 1;
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);
        let v_left = keys
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let mut idx = bisect::bisect_left(&v_left, &key)?;
        let len_keys = keys.len();
        let mut len_sublist = v_left.len()?;

        loop {
            if keys.get_item(pos)?.get_item(idx)?.ne(&key)? {
                return is_not_in_list_err(value);
            }
            if lists.get_item(pos)?.get_item(idx)?.eq(&value)? {
                let loc = self.loc(py, pos, idx as isize)?;
                if start <= loc && loc <= stop {
                    return Ok(loc);
                } else if loc > stop {
                    break;
                }
            }
            idx += 1;
            if idx == len_sublist {
                pos += 1;
                if pos == len_keys {
                    return is_not_in_list_err(value);
                }
                len_sublist = keys.get_item(pos)?.len()?;
                idx = 0;
            }
        }

        is_not_in_list_err(value)
    }
    fn update(&mut self, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let lists = self.lists.get().inner.clone_ref(py).into_bound(py);
        let maxes = self.maxes.get().inner.clone_ref(py).into_bound(py);
        let key_fn = &self.key.clone_ref(py).into_bound(py);
        let keys = self.keys.get().inner.clone_ref(py).into_bound(py);
        let mut values = iterable
            .try_iter()
            .and_then(|x| pylibs::builtins::sorted_by(&x, false, key_fn))?;

        if !maxes.is_empty() {
            if values.len() * 4 >= self.len {
                lists.append(values.into_pyochain()?)?;
                values = self
                    .collapse_lists(py)?
                    .get()
                    .inner
                    .clone_ref(py)
                    .into_bound(py);
                values.sort_by(key_fn, false)?;
                self.clear(py);
            } else {
                for val in values {
                    self.add(val)?;
                }
                return Ok(());
            }
        }

        let load = self.load;
        let new_maxes = (0..values.len())
            .step_by(load)
            .map(|pos| values.get_slice(pos, pos + load).into_pyochain())
            .try_fold(lists, try_iterator_into_list)?
            .iter()
            .map(|list_| {
                unsafe { list_.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .iter()
                    .map(|x| key_fn.call1((x,)))
                    .try_fold(PyList::empty(py), |acc, x| {
                        acc.append(x?)?;
                        Ok::<_, PyErr>(acc)
                    })?
                    .into_pyochain()
            })
            .try_fold(keys, try_iterator_into_list)?
            .iter()
            .map(|x| {
                unsafe { x.cast_into_unchecked::<PyoVec>() }
                    .get()
                    .inner
                    .bind(py)
                    .last()
            })
            .try_fold(PyList::empty(py), try_iterator_into_list)?;
        maxes.iadd(&new_maxes)?;
        self.len = values.len();
        self.idx.get().clear(py);
        Ok(())
    }
}

#[pymethods]
impl InnerKeyLists {
    fn bisect_key_left(&mut self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = key.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_left(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            Ok(self.len as isize)
        } else {
            let v = self
                .keys
                .bind(py)
                .get_item(pos)
                .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
            let idx = bisect::bisect_left(&v, &key)?;

            self.loc(py, pos, idx as isize)
        }
    }
    fn bisect_key_right(&mut self, key: Bound<'_, PyAny>) -> PyResult<isize> {
        let py = key.py();
        let maxes = self.maxes.get().inner.bind(py);

        if maxes.is_empty() {
            return Ok(0);
        }

        let pos = bisect::bisect_right(self.maxes.bind(py), &key)?;

        if pos == maxes.len() {
            return Ok(self.len as isize);
        }
        let v = &self
            .keys
            .get()
            .inner
            .bind(py)
            .get_item(pos)
            .map(|x| unsafe { x.cast_into_unchecked::<PyoVec>() })?;
        let idx = bisect::bisect_right(&v, &key)?;

        return self.loc(py, pos, idx as isize);
    }
}
/// Module for bisect functions, adapted from the Python standard library's bisect module.\
/// Adapted to only handle `pyochain::PyoVec` for both simplicity and performance.
pub mod bisect {
    use super::*;

    /// The following documentation and code is adapted from the Python standard library's bisect module.
    ///Return the index where to insert item x in list a, assuming a is sorted.

    ///The return value i is such that all e in a[:i] have e <= x, and all e in
    ///a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    ///insert just after the rightmost x already there.

    ///Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.
    #[pyfunction]
    pub fn bisect_right(vec: &Bound<'_, PyoVec>, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let lst = vec.get().inner.bind(vec.py());
        let mut high = lst.len();
        let mut lo = 0;
        // Note, the comparison uses "<" to match the
        // __lt__() logic in list.sort() and in heapq.
        while lo < high {
            let mid = (lo + high) / 2;
            if item.lt(lst.get_item(mid)?)? {
                high = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }
    /// The following documentation and code is adapted from the Python standard library's bisect module.\
    /// Return the index where to insert item x in list a, assuming a is sorted.\
    /// The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.\
    /// So if x already appears in the list, a.insert(i, x) will insert just before the leftmost x already there.\
    /// Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.\
    #[pyfunction]
    pub fn bisect_left(vec: &Bound<'_, PyoVec>, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let lst = vec.get().inner.bind(vec.py());
        let mut hi = lst.len();
        let mut lo = 0;
        // Note, the comparison uses "<" to match the
        // __lt__() logic in list.sort() and in heapq.
        while lo < hi {
            let mid = (lo + hi) / 2;
            if lst.get_item(mid)?.lt(item)? {
                lo = mid + 1;
            } else {
                hi = mid
            }
        }
        Ok(lo)
    }
}
#[inline]
fn not_in_list_err<T>(value: Bound<'_, PyAny>) -> PyResult<T> {
    let msg = format!("{} not in list", value.repr()?);
    Err(PyValueError::new_err(msg))
}
#[inline]
fn is_not_in_list_err<T>(value: Bound<'_, PyAny>) -> PyResult<T> {
    let msg = format!("{} is not in list", value.repr()?);
    Err(PyValueError::new_err(msg))
}
#[inline]
fn out_of_range_err<T>() -> PyResult<T> {
    let msg = "list index out of range";
    Err(PyIndexError::new_err(msg))
}
#[inline(always)]
fn iterator_into_list<'py, T: Sized + IntoPyObject<'py>>(
    acc: Bound<'py, PyList>,
    x: T,
) -> PyResult<Bound<'py, PyList>> {
    acc.append(x)?;
    Ok(acc)
}
#[inline(always)]
fn try_iterator_into_list<'py, T: Sized + IntoPyObject<'py>>(
    acc: Bound<'py, PyList>,
    x: PyResult<T>,
) -> PyResult<Bound<'py, PyList>> {
    acc.append(x?)?;
    Ok(acc)
}
