use crate::{
    abc,
    collections::{
        InnerKeyLists, InnerLists,
        sorted::{errors, iter},
    },
    iterators,
    pyovec::PyoVec,
    traits::IntoPyochain,
};
use either::Either;
use pyo3::{
    PyClass,
    exceptions::PyIndexError,
    prelude::*,
    types::{PyList, PyNotImplemented, PySequence, PySlice, PySliceIndices},
};
use pyochain_macros::py_abc;
use std::{
    cmp::Ordering,
    sync::{Mutex, MutexGuard, TryLockError, atomic::Ordering as AtomicOrdering},
};
use tap::prelude::*;

pub const DEFAULT_LOAD_FACTOR: usize = 1000;
pub type BoolOrNotImpl<'py> = PyResult<Either<bool, Bound<'py, PyNotImplemented>>>;
pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;

pub(super) struct ListsData {
    pub(super) lists: Vec<Vec<Py<PyAny>>>,
    pub(super) maxes: Vec<Py<PyAny>>,
    pub(super) idx: Vec<usize>,
}
impl ListsData {
    pub fn new() -> Self {
        Self {
            lists: Vec::new(),
            maxes: Vec::new(),
            idx: Vec::new(),
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
}

pub(super) fn try_lock_recover<'a, T>(mutex: &'a Mutex<T>, msg: &str) -> MutexGuard<'a, T> {
    match mutex.try_lock() {
        Ok(guard) => guard,
        //Recover if the guard was poisoned by an earlier panic instead of cascading.
        Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
        Err(TryLockError::WouldBlock) => panic!("{msg}"),
    }
}

pub trait RustGetters:
    Sized + PyClass + PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn get_data(&self) -> MutexGuard<'_, ListsData>;
    fn get_offset(&self) -> usize;
    fn set_offset(&self, offset: usize);
    fn set_load(&self, load: usize);
}
macro_rules! impl_rs_getters {
    ($t:ty) => {
        impl RustGetters for $t {
            #[inline(always)]
            fn get_data(&self) -> MutexGuard<'_, ListsData> {
                try_lock_recover(&self.data, "data already locked - reentrant bug")
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
            fn set_load(&self, load: usize) {
                self.load.store(load, AtomicOrdering::Relaxed);
            }
        }
    };
}
impl_rs_getters!(InnerLists);
impl_rs_getters!(InnerKeyLists);
#[py_abc(InnerLists, InnerKeyLists)]
pub(super) trait InnerSortedGetters: RustGetters {
    #[getter]
    fn get_load(&self) -> usize;
    #[getter]
    fn get_len(&self) -> usize;
    #[setter]
    fn set_len(&self, len: usize);
    fn eq<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn ne<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn lt<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn gt<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn le<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
    fn ge<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py>;
}
macro_rules! impl_inner_sorted_rs {
    ($t:ty) => {
        impl InnerSortedGetters for $t {
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

            fn eq<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        if self.get_len().ne(&seq.len()?) {
                            Either::Left(false).pipe(Ok)
                        } else {
                            let py = seq.py();
                            self.get_data()
                                .lists
                                .iter()
                                .flat_map(move |x| x.iter())
                                .zip(seq.try_iter()?)
                                .map(|(a, b)| a.bind(py).eq(b?))
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
            fn ne<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        if self.get_len().ne(&seq.len()?) {
                            Either::Left(true).pipe(Ok)
                        } else {
                            let py = seq.py();
                            self.get_data()
                                .lists
                                .iter()
                                .flat_map(move |x| x.iter())
                                .zip(seq.try_iter()?)
                                .map(|(a, b)| a.bind(py).eq(b?))
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

            fn lt<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        let py = seq.py();
                        for (alpha, beta) in self
                            .get_data()
                            .lists
                            .iter()
                            .flat_map(move |x| x.iter())
                            .zip(seq.try_iter()?)
                        {
                            let a = alpha.bind(py);
                            let b = beta?;
                            if a.ne(&b)? {
                                return a.lt(&b).map(Either::Left);
                            }
                        }

                        self.get_len().lt(&seq.len()?).pipe(Either::Left).pipe(Ok)
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn gt<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        let py = seq.py();
                        for (alpha, beta) in self
                            .get_data()
                            .lists
                            .iter()
                            .flat_map(move |x| x.iter())
                            .zip(seq.try_iter()?)
                        {
                            let b = beta?;
                            let a = alpha.bind(py);
                            if a.ne(&b)? {
                                return Either::Left(a.gt(&b)?).pipe(Ok);
                            }
                        }
                        self.get_len().gt(&seq.len()?).pipe(Either::Left).pipe(Ok)
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn le<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        let py = seq.py();
                        for (alpha, beta) in self
                            .get_data()
                            .lists
                            .iter()
                            .flat_map(move |x| x.iter())
                            .zip(seq.try_iter()?)
                        {
                            let b = beta?;
                            let a = alpha.bind(py);
                            if a.ne(&b)? {
                                return a.le(b).map(Either::Left);
                            }
                        }

                        self.get_len().le(&seq.len()?).pipe(Either::Left).pipe(Ok)
                    }

                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }

            fn ge<'py>(&self, other: SeqOrAny<'py>) -> BoolOrNotImpl<'py> {
                match other {
                    Either::Left(seq) => {
                        let py = seq.py();
                        for (alpha, beta) in self
                            .get_data()
                            .lists
                            .iter()
                            .flat_map(move |x| x.iter())
                            .zip(seq.try_iter()?)
                        {
                            let b = beta?;
                            let a = alpha.bind(py);
                            if a.ne(&b)? {
                                return a.ge(b).map(Either::Left);
                            }
                        }

                        self.get_len().ge(&seq.len()?).pipe(Either::Left).pipe(Ok)
                    }
                    Either::Right(any) => errors::not_impl(any.py()),
                }
            }
        }
    };
}
impl_inner_sorted_rs!(InnerLists);
impl_inner_sorted_rs!(InnerKeyLists);
#[py_abc(InnerLists, InnerKeyLists)]
pub(super) trait InnerSorted: InnerSortedGetters {
    #[skip]
    fn wrap_iter<'py>(
        py: Python<'py>,
        inner: iter::BoundedIter<Self>,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    fn bisect_left(&self, value: Bound<'_, PyAny>) -> PyResult<isize>;
    fn bisect_right(&self, value: &Bound<'_, PyAny>) -> PyResult<isize>;
    fn clear(&self) -> ();
    fn contains(&self, value: Bound<'_, PyAny>) -> PyResult<bool>;
    #[skip]
    fn delete(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
        idx: usize,
    ) -> PyResult<()>;
    #[skip]
    fn expand(
        &self,
        py: Python<'_>,
        data: &mut MutexGuard<'_, ListsData>,
        pos: usize,
    ) -> PyResult<()>;
    fn add(&self, py: Python<'_>, value: Py<PyAny>) -> PyResult<()>;
    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize>;
    fn update(&self, iterable: &Bound<'_, PyAny>) -> PyResult<()>;
    fn mul<'py>(&self, py: Python<'py>, num: usize) -> PyResult<Bound<'py, PyoVec>> {
        let values = self.collapse_lists(py);
        (0..num)
            .flat_map(|_| values.iter())
            .pipe(|x| PyList::new(py, x))?
            .into_pyochain()
    }
    #[pyo3(signature = (minimum = None, maximum = None, inclusive = (true, true), *, reverse = false))]
    fn irange<'py>(
        slf: Bound<'py, Self>,
        minimum: Option<Bound<'py, PyAny>>,
        maximum: Option<Bound<'py, PyAny>>,
        inclusive: (bool, bool),
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>>;
    #[skip]
    fn update_from_vec(&self, py: Python<'_>, iterable: Vec<Py<PyAny>>) -> PyResult<()>;

    fn reset(&self, py: Python<'_>, load: usize) -> PyResult<()> {
        let values = self.collapse_lists(py);
        self.clear();
        self.set_load(load);
        self.update_from_vec(py, values)
    }
    #[pyo3(signature = (value, start = None, stop = None))]
    fn index(
        &self,
        value: Bound<'_, PyAny>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<isize>;
    fn collapse_lists<'py>(&self, py: Python<'py>) -> Vec<Py<PyAny>> {
        self.get_data().collapse(py)
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
    #[skip]
    fn build_index(&self, data: &mut ListsData) -> PyResult<()> {
        let row0 = data.lists.iter().map(|x| x.len()).collect::<Vec<usize>>();

        if row0.len() == 1 {
            data.idx.extend(row0);
            self.set_offset(0);
            return Ok(());
        }

        let mut row1 = row0
            .chunks(2)
            .map(|pair| pair.iter().sum())
            .collect::<Vec<usize>>();

        if row1.len() == 1 {
            let combined = row1.into_iter().chain(row0);
            data.idx.clear();
            data.idx.extend(combined);
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
        data.idx.extend(flat);
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
    #[skip]
    fn loc(
        &self,
        data: &mut MutexGuard<'_, ListsData>,
        mut pos: usize,
        idx: isize,
    ) -> PyResult<isize> {
        if pos == 0 {
            Ok(idx)
        } else {
            if data.idx.is_empty() {
                self.build_index(data)?;
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            pos += self.get_offset();
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
    #[skip]
    fn pos(&self, data: &mut ListsData, mut idx: isize) -> PyResult<(usize, isize)> {
        if idx < 0 {
            if (-idx) <= data.lists.last().unwrap().len() as isize {
                return Ok((
                    data.lists.len() - 1,
                    data.lists.last().unwrap().len() as isize + idx,
                ));
            }

            idx += self.get_len() as isize;

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= self.get_len() as isize {
            return errors::out_of_range_err();
        }

        if idx < data.lists[0].len() as isize {
            return Ok((0, idx));
        }

        if data.idx.is_empty() {
            self.build_index(data)?;
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

        return Ok((pos - self.get_offset(), idx));
    }
    #[pyo3(signature = (index = -1))]
    fn pop<'py>(&self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        if self.get_len() == 0 {
            let msg = "pop index out of range";
            return Err(PyIndexError::new_err(msg));
        }

        let (pos, idx) = {
            let mut data = self.get_data();
            let len_last = data.lists.last().unwrap().len() as isize;
            match index {
                0 => (0, 0),
                -1 => {
                    let pos = data.lists.len() - 1;
                    (pos, data.lists[pos].len() - 1)
                }
                _ if 0 <= index && index < data.lists[0].len() as isize => (0, index as usize),
                _ if -len_last < index && index < 0 => {
                    let pos = data.lists.len() - 1;
                    (pos, (len_last + index) as usize)
                }
                _ => {
                    let (pos, idx) = self.pos(&mut data, index)?;
                    (pos, idx as usize)
                }
            }
        };
        let mut data = self.get_data();
        let val = data.lists[pos][idx].clone_ref(py);
        self.delete(py, &mut data, pos, idx)?;
        Ok(val.into_bound(py))
    }
    //TODO: Refactor this in a new module. get_item_from_slice is way too long and complex.
    fn getitem<'py>(
        &self,
        py: Python<'py>,
        index: Either<isize, Bound<'py, PySlice>>,
    ) -> PyResult<Either<Bound<'py, PyAny>, Bound<'py, PyoVec>>> {
        let mut data = self.get_data();
        match index {
            Either::Right(slice) => self
                .getitem_from_slice(py, &mut data, slice)?
                .iter()
                .pipe(|elements| PyList::new(py, elements))?
                .into_pyochain()
                .map(Either::Right),
            Either::Left(index) => self
                .getitem_from_int(py, &mut data, index)
                .map(Either::Left),
        }
    }
    #[skip]
    fn getitem_from_int<'py>(
        &self,
        py: Python<'py>,
        data: &mut MutexGuard<'_, ListsData>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slf_len = self.get_len();
        let len_last = data
            .lists
            .last()
            .ok_or(PyIndexError::new_err("list index out of range"))?
            .len() as isize;
        match (index, slf_len != 0) {
            (0, true) => data.lists[0][0].clone_ref(py).into_bound(py).pipe(Ok),
            (-1, true) => data
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
            (_, true) if 0 <= index && index < data.lists[0].len() as isize => data.lists[0]
                [index as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, true) if -len_last < index && index < 0 => data.lists.last().unwrap()
                [(len_last + index) as usize]
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            _ => {
                let (pos, idx) = self.pos(data, index)?;
                data.lists[pos][idx as usize]
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    #[skip]
    fn getitem_from_slice<'py>(
        &self,
        py: Python<'py>,
        data: &mut MutexGuard<'_, ListsData>,
        slice: Bound<'py, PySlice>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let slice_result = |data: &MutexGuard<'_, ListsData>,
                            start_pos: usize,
                            stop_pos: usize,
                            start_idx: usize,
                            stop_idx: usize| {
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
        };

        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.get_len() as isize)?;
        let stop_eq_len = stop == self.get_len() as isize;
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => data.collapse(py).pipe(Ok),
            (1, Ordering::Less) => {
                let (start_pos, start_idx) = self.pos(data, start)?;
                let start_list = &data.lists[start_pos];
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
                        let stop_pos = data.lists.len() - 1;
                        let stop_idx = data.lists[stop_pos].len();
                        slice_result(&data, start_pos, stop_pos, start_idx as usize, stop_idx)
                    }
                    (false, false) => {
                        let (stop_pos, stop_idx) = self.pos(data, stop)?;
                        slice_result(
                            &data,
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
                    self.getitem_from_slice(py, data, PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            _ if step > 0 => (start..stop)
                .step_by(step as usize)
                .map(|i| self.getitem_from_int(py, data, i).map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>(),
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (_, Ordering::Less | Ordering::Equal) => Ok(Vec::new()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.getitem_from_int(py, data, i).map(Bound::unbind))
                    .collect::<PyResult<Vec<_>>>()
            }
        }
    }

    fn delitem(&self, py: Python<'_>, index: Either<isize, Bound<'_, PySlice>>) -> PyResult<()> {
        match index {
            Either::Right(slice) => self.delitem_from_slice(py, slice),
            Either::Left(index) => {
                let mut data = self.get_data();
                let (pos, idx) = self.pos(&mut data, index)?;
                self.delete(py, &mut data, pos, idx as usize)
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
                self.clear();
                Ok(())
            }
            (1, Ordering::Less) if length <= 8 * (stop - start) => {
                let mut data = self.get_data();
                let mut values =
                    self.getitem_from_slice(py, &mut data, PySlice::new(py, 0, start, 1))?;
                if stop < length {
                    let new_slice =
                        self.getitem_from_slice(py, &mut data, PySlice::new(py, stop, length, 1))?;
                    values.extend(new_slice);
                }
                drop(data);
                self.clear();
                self.update_from_vec(py, values)?;
                Ok(())
            }
            _ if step > 0 => {
                let mut data = self.get_data();
                (start..stop)
                    .step_by(step as usize)
                    .rev()
                    .try_for_each(|idx| {
                        let (pos, idx) = self.pos(&mut data, idx)?;
                        self.delete(py, &mut data, pos, idx as usize)
                    })
            }
            // Negative step with nothing to delete (mirrors Python's
            // `range`, which is empty when `start <= stop`).
            (_, Ordering::Less | Ordering::Equal) => Ok(()),
            _ => {
                let mut data = self.get_data();
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .try_for_each(|idx| {
                        let (pos, idx) = self.pos(&mut data, idx)?;
                        self.delete(py, &mut data, pos, idx as usize)
                    })
            }
        }
    }
    #[pyo3(signature = (start = None, stop = None, *, reverse = false))]
    fn islice<'py>(
        slf: Bound<'py, Self>,
        py: Python<'py>,
        start: Option<isize>,
        stop: Option<isize>,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        match slf.get().islice_specs(py, start, stop)? {
            None => iterators::Iter::empty(py)?.into_super().pipe(Ok),
            Some(bounds) => Self::islice_iter(slf, bounds, reverse),
        }
    }

    #[skip]
    fn islice_specs(
        &self,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<iter::IsliceBounds>> {
        let length = self.get_len() as isize;

        if length == 0 {
            return Ok(None);
        }
        //NOTE: Need to investiguate why we need to use PySlice at all. Same pattern in SliceView original code.
        let indices =
            PySlice::new(py, start.unwrap_or(0), stop.unwrap_or(length), 1).indices(length)?;

        if indices.start >= indices.stop {
            Ok(None)
        } else {
            let mut data = self.get_data();
            let (min_pos, min_idx) = self.pos(&mut data, indices.start)?;

            let (max_pos, max_idx) = if indices.stop == length {
                (
                    data.lists.len() - 1,
                    data.lists.last().unwrap().len() as isize,
                )
            } else {
                self.pos(&mut data, indices.stop)?
            };

            Ok(Some(iter::IsliceBounds::new(
                min_pos,
                min_idx as usize,
                max_pos,
                max_idx as usize,
            )))
        }
    }

    /// Return an iterator that slices sorted list using two index pairs.\
    /// The index pairs are (min_pos, min_idx) and (max_pos, max_idx), the first inclusive and the latter exclusive.\
    /// See `_pos` for details on how an index is converted to an index pair.\
    /// When `reverse` is `True`, values are yielded from the iterator in reverse order.
    #[skip]
    fn islice_iter<'py>(
        slf: Bound<'py, Self>,
        bounds: iter::IsliceBounds,
        reverse: bool,
    ) -> PyResult<Bound<'py, abc::PyoIterator>> {
        let py = slf.py();
        let dir = if reverse {
            iter::Dir::Bwd
        } else {
            iter::Dir::Fwd
        };
        Self::wrap_iter(py, iter::BoundedIter::new(slf.unbind(), bounds, dir))
    }
    fn reversed(slf: Bound<'_, Self>) -> PyResult<Bound<'_, abc::PyoIterator>> {
        let py = slf.py();
        Self::wrap_iter(py, iter::BoundedIter::full(slf.unbind(), iter::Dir::Bwd))
    }

    fn iter(slf: Bound<'_, Self>) -> PyResult<Bound<'_, abc::PyoIterator>> {
        let py = slf.py();
        Self::wrap_iter(py, iter::BoundedIter::full(slf.unbind(), iter::Dir::Fwd))
    }
}
