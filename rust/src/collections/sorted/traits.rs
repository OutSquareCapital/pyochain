use crate::collections::sorted::errors;
use crate::collections::sorted::iter::try_iterator_into_list;
use crate::collections::{InnerKeyLists, InnerLists};
use crate::pyo3_ext::prelude::*;
use crate::seq::{IntoPyochain, PyoVec};
use either::Either;
use pyo3::PyClass;
use pyo3::exceptions::PyIndexError;
use pyo3::types::{PyNotImplemented, PySequence, PySlice, PySliceIndices};
use pyo3::{prelude::*, types::PyList};
use pyochain_macros::py_abc;
use std::cmp::Ordering;
use std::sync::atomic::Ordering as AtomicOrdering;
use tap::prelude::*;

pub const DEFAULT_LOAD_FACTOR: usize = 1000;
pub type BoolOrNotImpl<'py> = PyResult<Either<bool, Bound<'py, PyNotImplemented>>>;
pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;

pub trait RustGetters:
    Sized + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn get_idx(&self) -> std::sync::MutexGuard<'_, Vec<usize>>;
}
macro_rules! impl_rs_getters {
    ($t:ty) => {
        impl RustGetters for $t {
            #[inline(always)]
            fn get_idx(&self) -> std::sync::MutexGuard<'_, Vec<usize>> {
                self.idx.lock().unwrap()
            }
        }
    };
}
impl_rs_getters!(InnerLists);
impl_rs_getters!(InnerKeyLists);
#[py_abc(InnerLists, InnerKeyLists)]
pub(super) trait InnerSortedGetters: RustGetters {
    #[getter]
    fn get_lists(&self, py: Python<'_>) -> Py<PyoVec>;
    #[getter]
    fn get_maxes(&self, py: Python<'_>) -> Py<PyList>;
    #[getter]
    fn get_load(&self) -> usize;
    #[getter]
    fn get_offset(&self) -> usize;
    #[getter]
    fn get_len(&self) -> usize;
    fn set_offset(&self, offset: usize);
    #[setter]
    fn set_len(&self, len: usize);
    fn set_load(&self, load: usize);
    fn eq<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn ne<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn lt<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn gt<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn le<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn ge<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl InnerSortedGetters for $t {
            #[inline(always)]
            fn get_lists(&self, py: Python<'_>) -> Py<PyoVec> {
                self.lists.clone_ref(py)
            }
            #[inline(always)]
            fn get_offset(&self) -> usize {
                self.offset.load(AtomicOrdering::Relaxed)
            }
            #[inline(always)]
            fn set_offset(&self, offset: usize) {
                self.offset.store(offset, AtomicOrdering::Relaxed);
            }
            #[inline(always)]
            fn get_len(&self) -> usize {
                self.len.load(AtomicOrdering::Relaxed)
            }
            #[inline(always)]
            fn set_len(&self, len: usize) {
                self.len.store(len, AtomicOrdering::Relaxed);
            }
            #[inline(always)]
            fn get_load(&self) -> usize {
                self.load.load(AtomicOrdering::Relaxed)
            }
            #[inline(always)]
            fn set_load(&self, load: usize) {
                self.load.store(load, AtomicOrdering::Relaxed);
            }
            #[inline(always)]
            fn get_maxes(&self, py: Python<'_>) -> Py<PyList> {
                self.maxes.clone_ref(py)
            }

            fn eq<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        if slf.get().get_len().ne(&seq.len()?) {
                            Either::Left(false).pipe(Ok)
                        } else {
                            slf.iter()
                                .zip(seq.try_iter()?)
                                .map(|(a, b)| a.eq(b?))
                                .find_map(|x| match x {
                                    Ok(true) => None,
                                    Ok(false) => Some(Ok(false)),
                                    Err(e) => Some(Err(e)),
                                })
                                .unwrap_or(Ok(true))
                                .map(Either::Left)
                        }
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }
            fn ne<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        if slf.get().get_len().ne(&seq.len()?) {
                            Either::Left(true).pipe(Ok)
                        } else {
                            slf.iter()
                                .zip(seq.try_iter()?)
                                .map(|(a, b)| a.eq(b?))
                                .find_map(|x| match x {
                                    Ok(true) => None,
                                    Ok(false) => Some(Ok(true)),
                                    Err(e) => Some(Err(e)),
                                })
                                .unwrap_or(Ok(false))
                                .map(Either::Left)
                        }
                    }
                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn lt<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        for (alpha, beta) in slf.iter().zip(seq.try_iter()?) {
                            let b = beta?;
                            if alpha.ne(&b)? {
                                return alpha.lt(&b).map(Either::Left);
                            }
                        }

                        return slf
                            .get()
                            .get_len()
                            .lt(&seq.len()?)
                            .pipe(Either::Left)
                            .pipe(Ok);
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn gt<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        for (alpha, beta) in slf.iter().zip(seq.try_iter()?) {
                            let b = beta?;
                            if alpha.ne(&b)? {
                                return Either::Left(alpha.gt(b)?).pipe(Ok);
                            }
                        }
                        slf.get()
                            .get_len()
                            .gt(&seq.len()?)
                            .pipe(Either::Left)
                            .pipe(Ok)
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn le<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        for (alpha, beta) in slf.iter().zip(seq.try_iter()?) {
                            let b = beta?;
                            if alpha.ne(&b)? {
                                return alpha.le(b).map(Either::Left);
                            }
                        }

                        slf.get()
                            .get_len()
                            .le(&seq.len()?)
                            .pipe(Either::Left)
                            .pipe(Ok)
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn ge<'py>(slf: Bound<'py, Self>, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        for (alpha, beta) in slf.iter().zip(seq.try_iter()?) {
                            let b = beta?;
                            if alpha.ne(&b)? {
                                return alpha.ge(b).map(Either::Left);
                            }
                        }

                        slf.get()
                            .get_len()
                            .ge(&seq.len()?)
                            .pipe(Either::Left)
                            .pipe(Ok)
                    }
                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }
        }
    };
}
impl_inner_sorted_rs!(InnerLists);
impl_inner_sorted_rs!(InnerKeyLists);
pub(super) trait InnerSortedIter<'py>: Sized {
    fn iter(&self) -> impl Iterator<Item = Bound<'py, PyAny>>;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl<'py> InnerSortedIter<'py> for Bound<'py, $t> {
            fn iter(&self) -> impl Iterator<Item = Bound<'py, PyAny>> {
                let py = self.py();
                self.get()
                    .get_lists(py)
                    .get()
                    .inner
                    .bind(py)
                    .iter()
                    .flat_map(move |x| {
                        unsafe { x.cast_unchecked::<PyoVec>() }
                            .get()
                            .inner
                            .bind(py)
                            .iter()
                    })
            }
        }
    };
}
impl_inner_sorted_rs!(InnerLists);
impl_inner_sorted_rs!(InnerKeyLists);

#[py_abc(InnerLists, InnerKeyLists)]
pub(super) trait InnerSorted: InnerSortedGetters {
    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    fn clear(&self, py: Python<'_>) -> ();
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool>;
    fn delete(&self, py: Python<'_>, pos: usize, idx: usize) -> PyResult<()>;
    fn expand(&self, py: Python<'_>, pos: usize) -> PyResult<()>;
    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.collapse_lists(py)?.into_any();
        self.clear(py);
        self.set_load(load);
        self.update(&values)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
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
    fn build_index(&self, py: Python<'_>) -> PyResult<()> {
        let mut idx = self.get_idx();
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);

        let row0 = lists
            .iter()
            .map(|x| x.len())
            .collect::<PyResult<Vec<usize>>>()?;

        if row0.len() == 1 {
            idx.extend(row0);
            self.set_offset(0);
            return Ok(());
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            idx.clear();
            idx.extend(combined);
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

        let flat = tree.into_iter().rev().flatten();
        idx.extend(flat);
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
    fn loc(&self, py: Python<'_>, mut pos: usize, idx: isize) -> PyResult<isize> {
        if pos == 0 {
            Ok(idx)
        } else {
            if self.get_idx().is_empty() {
                self.build_index(py)?;
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            pos += self.get_offset();
            // Iterate until reaching the root of the index tree at pos = 0.
            let total = self.get_idx().pipe_ref_mut(|idx| {
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
    fn pos(&self, py: Python<'_>, mut idx: isize) -> PyResult<(usize, isize)> {
        let lists = self.get_lists(py).get().inner.clone_ref(py).into_bound(py);
        if idx < 0 {
            if (-idx) <= lists.last()?.len()? as isize {
                return Ok((lists.len() - 1, lists.last()?.len()? as isize + idx));
            }

            idx += self.get_len() as isize;

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= self.get_len() as isize {
            return errors::out_of_range_err();
        }

        if idx < lists.get_item(0)?.len()? as isize {
            return Ok((0, idx));
        }

        if self.get_idx().is_empty() {
            self.build_index(py)?;
        }
        let pos = self.get_idx().pipe_ref_mut(|index| {
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

        return Ok((pos - self.get_offset(), idx));
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
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
        &self,
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
        &self,
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
        &self,
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

    fn delitem(&self, py: Python<'_>, index: Either<isize, Bound<'_, PySlice>>) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.delitem_from_slice(py, slice),
            Either::Left(index) => {
                let (pos, idx) = self.pos(py, index)?;
                self.delete(py, pos, idx as usize)
            }
        }
    }
    fn delitem_from_slice(&self, py: Python<'_>, slice: Bound<'_, PySlice>) -> PyResult<()> {
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
