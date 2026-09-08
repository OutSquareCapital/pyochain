use std::cmp::Ordering;

use either::Either;
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyNotImplemented, PySequence, PySlice, PySliceIndices},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use tap::Pipe;

use crate::{Bounds, Loc, debug::check_list, errors, pyassert, traits::NestedVec};

/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;
pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;
pub struct InnerData {
    pub lists: Vec<VecPy>,
    pub maxes: VecPy,
    pub idx: Vec<usize>,
    pub len: usize,
    pub offset: usize,
    pub load: usize,
}

impl Default for InnerData {
    fn default() -> Self {
        Self {
            lists: Vec::default(),
            maxes: Vec::default(),
            idx: Vec::default(),
            len: usize::default(),
            offset: usize::default(),
            load: 1000,
        }
    }
}
impl InnerData {
    #[inline]
    pub fn clear(&mut self) {
        self.lists.clear();
        self.maxes.clear();
        self.idx.clear();
        self.len = 0;
        self.offset = 0;
    }
    #[inline]
    #[must_use]
    pub fn collapse(&self, py: Python<'_>) -> VecPy {
        self.iter().map(|x| x.clone_ref(py)).collect()
    }
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &Py<PyAny>> {
        self.lists.iter().flatten()
    }
    #[inline]
    pub fn concat(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<VecPy> {
        self.iter()
            .map(|x| x.clone_ref(py).pipe(Ok))
            .chain(other.try_iter()?.map(|x| x?.unbind().pipe(Ok)))
            .collect()
    }
    pub fn check(&self, py: Python<'_>) -> PyResult<()> {
        check_list(self, py)
    }
    pub fn loc(&mut self, loc: &Loc) -> usize {
        if loc.pos == 0 {
            loc.idx
        } else {
            if self.idx.is_empty() {
                self.build_index();
            }
            // Increment pos to point in the index to len(self.lists[pos]).
            let mut i = loc.pos + self.offset;
            // Iterate until reaching the root of the index tree at pos = 0.
            let total = self.idx.pipe_ref_mut(|idx| {
                let mut total = 0;
                while i != 0 {
                    // Right-child nodes are at even indices. At such indices
                    // account the total below the left child node.

                    if i.is_multiple_of(2) {
                        total += idx[i - 1];
                    }

                    // Advance pos to the parent node.

                    i = (i - 1) >> 1;
                }
                total
            });

            total + loc.idx
        }
    }

    pub fn check_empty(&self) -> PyResult<()> {
        pyassert!(self.len == 0);
        pyassert!(self.maxes.is_empty());
        pyassert!(self.lists.is_empty());
        Ok(())
    }
    #[inline]
    #[must_use]
    pub fn repeat(&self, py: Python<'_>, num: usize) -> VecPy {
        let values = self.collapse(py);
        (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect()
    }

    pub fn get_item<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let mut bounds = Bounds::default();
        let len_last = self
            .lists
            .last()
            .ok_or(PyIndexError::new_err("list index out of range"))?
            .len()
            .cast_signed();
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
            (_, true) if 0 <= index && index < self.lists[0].len().cast_signed() => self.lists[0]
                [index.cast_unsigned()]
            .clone_ref(py)
            .into_bound(py)
            .pipe(Ok),
            (_, true) if -len_last < index && index < 0 => self.lists.last().unwrap()
                [(len_last + index).cast_unsigned()]
            .clone_ref(py)
            .into_bound(py)
            .pipe(Ok),
            _ => {
                self.set_pos(index, &mut bounds.min)?;
                self.lists
                    .loc(&bounds.min)
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    pub fn get_slice<'py>(
        &mut self,
        py: Python<'py>,
        slice: &Bound<'py, PySlice>,
    ) -> PyResult<VecPy> {
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.len.cast_signed())?;
        let stop_eq_len = stop == self.len.cast_signed();
        let mut bounds = Bounds::default();
        match (step, start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (1, Ordering::Less) if start == 0 && stop_eq_len => self.collapse(py).pipe(Ok),
            (1, Ordering::Less) => {
                self.set_pos(start, &mut bounds.min)?;
                let start_list = &self.lists[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (stop - start).cast_unsigned();
                match (start_list.len() >= bounds.max.idx, stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (true, _) => start_list[bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>()
                        .pipe(Ok),
                    (false, true) => {
                        bounds.max.pos = self.lists.len() - 1;
                        bounds.max.idx = self.lists.loc_len(&bounds.max);
                        get_slice(&self.lists, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                    (false, false) => {
                        self.set_pos(stop, &mut bounds.max)?;
                        get_slice(&self.lists, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                }
            }
            (-1, Ordering::Greater) => {
                let mut result = self.get_slice(py, &PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            _ if step > 0 => (start..stop)
                .step_by(step.cast_unsigned())
                .map(|i| self.get_item(py, i).map(Bound::unbind))
                .collect(),
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (_, Ordering::Less | Ordering::Equal) => Ok(Vec::new()),
            _ => {
                // Negative step, `start > stop` guaranteed by the arm above.
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.get_item(py, i).map(Bound::unbind))
                    .collect()
            }
        }
    }

    /// Build a positional index for indexing the sorted list.
    /// Indexes are represented as binary trees in a dense array notation similar to a binary heap.
    /// For example, given a lists representation storing integers:\
    ///     0: [1, 2, 3]
    ///     1: [4, 5]
    ///     2: [6, 7, 8, 9]
    ///     3: [10, 11, 12, 13, 14]
    /// The first transformation maps the sub-lists by their length.\
    /// The first row of the index is the length of the `sub-lists::`
    ///     0: [3, 2, 4, 5]
    /// Each row after that is the sum of consecutive pairs of the previous row:
    ///     1: [5, 9]
    ///     2: [14]
    /// Finally, the index is built by concatenating these lists together:
    ///     _index = [14, 5, 9, 3, 2, 4, 5]
    /// An offset storing the start of the first row is also stored:
    ///     _offset = 3
    /// When built, the index can be used for efficient indexing into the list.
    fn build_index(&mut self) {
        let row0 = self.lists.iter().map(Vec::len).collect::<Vec<usize>>();

        if row0.len() == 1 {
            self.idx.extend(&row0);
            self.offset = 1;
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
        }
    }
    pub(super) fn expand_on_empty_idx(&mut self, pos: usize) {
        let mut child = self.offset + pos;
        while child != 0 {
            self.idx[child] += 1;
            child = (child - 1) >> 1;
        }
        self.idx[0] += 1;
    }
    pub(super) fn remove_pos(&mut self, loc: &Loc) {
        self.lists.remove(loc.pos);
        self.maxes.remove(loc.pos);
        self.idx.clear();
    }
    pub(super) fn expand_at_pos(
        &mut self,
        pos: usize,
        half: VecPy,
        last_max: Py<PyAny>,
        new_max_at_pos: Py<PyAny>,
    ) {
        self.maxes[pos] = new_max_at_pos;
        self.maxes.insert(pos + 1, last_max);
        self.lists.insert(pos + 1, half);
        self.idx.clear();
    }

    pub(super) fn delete_on_idx(&mut self, loc: &Loc, max_at_pos: Py<PyAny>) {
        self.maxes[loc.pos] = max_at_pos;

        if !self.idx.is_empty() {
            let mut child = self.offset + loc.pos;
            while child > 0 {
                self.idx[child] -= 1;
                child = (child - 1) >> 1;
            }
            self.idx[0] -= 1;
        }
    }

    pub(super) fn set_pos(&mut self, mut idx: isize, loc: &mut Loc) -> PyResult<()> {
        if idx < 0 {
            if idx >= -self.lists.last().unwrap().len().cast_signed() {
                loc.pos = self.lists.len() - 1;
                loc.idx = (self.lists.last().unwrap().len().cast_signed() + idx).cast_unsigned();
                return Ok(());
            }

            idx += self.len.cast_signed();

            if idx < 0 {
                return errors::out_of_range_err();
            }
        } else if idx >= self.len.cast_signed() {
            return errors::out_of_range_err();
        }

        if idx < self.lists[0].len().cast_signed() {
            loc.pos = 0;
            loc.idx = idx.cast_unsigned();
            return Ok(());
        }

        if self.idx.is_empty() {
            self.build_index();
        }
        let pos = self.idx.pipe_ref_mut(|index| {
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

        loc.pos = pos - self.offset;
        loc.idx = idx.cast_unsigned();
        Ok(())
    }

    pub fn get_islice_specs(
        &mut self,
        py: Python<'_>,
        start: Option<isize>,
        stop: Option<isize>,
    ) -> PyResult<Option<Bounds>> {
        let length = self.len.cast_signed();
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
                self.set_pos(indices.start, &mut bounds.min)?;

                if indices.stop == length {
                    bounds.max.pos = self.lists.len() - 1;
                    bounds.max.idx = self.lists.last().unwrap().len();
                } else {
                    self.set_pos(indices.stop, &mut bounds.max)?;
                }

                Ok(Some(bounds))
            }
        }
    }

    pub fn eq<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                if self.len.ne(&seq.len()?) {
                    Either::Left(false).pipe(Ok)
                } else {
                    let py = seq.py();
                    self.iter()
                        .zip(seq.iter_py())
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

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    pub fn ne<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                if self.len.ne(&seq.len()?) {
                    Either::Left(true).pipe(Ok)
                } else {
                    let py = seq.py();
                    self.iter()
                        .zip(seq.iter_py())
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
            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    pub fn lt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                for (alpha, beta) in self.iter().zip(seq.iter_py()) {
                    let a = alpha.bind(py);
                    let b = beta?;
                    if a.ne(&b)? {
                        return a.lt(&b).map(Either::Left);
                    }
                }

                self.len.lt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    pub fn gt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                for (alpha, beta) in self.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return Either::Left(a.gt(&b)?).pipe(Ok);
                    }
                }
                self.len.gt(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    pub fn le<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                for (alpha, beta) in self.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.le(b).map(Either::Left);
                    }
                }

                self.len.le(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }

            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }

    pub fn ge<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        match other {
            Either::Left(seq) => {
                let py = seq.py();
                for (alpha, beta) in self.iter().zip(seq.iter_py()) {
                    let b = beta?;
                    let a = alpha.bind(py);
                    if a.ne(&b)? {
                        return a.ge(b).map(Either::Left);
                    }
                }

                self.len.ge(&seq.len()?).pipe(Either::Left).pipe(Ok)
            }
            Either::Right(any) => PyNotImplemented::from_cmp(any.py()),
        }
    }
    pub(super) fn extend_lists(&mut self, py: Python<'_>, values: &[Py<PyAny>]) {
        let val_len = values.len();
        (0..val_len)
            .step_by(self.load)
            .map(|pos| {
                values[pos..(pos + self.load).min(val_len)]
                    .iter()
                    .map(|x| x.clone_ref(py))
                    .collect::<Vec<_>>()
            })
            .pipe(|it| self.lists.extend(it));
    }
}
pub trait InnerGetter: Sized {
    fn inner(&self) -> &InnerData;
    fn inner_mut(&mut self) -> &mut InnerData;
}

pub trait ListDataGetters: Sized {
    fn lists(&self) -> &[VecPy];
    fn lists_mut(&mut self) -> &mut Vec<VecPy>;
    fn maxes(&self) -> &[Py<PyAny>];
    fn maxes_mut(&mut self) -> &mut VecPy;
    fn idx(&self) -> &[usize];
    fn idx_mut(&mut self) -> &mut Vec<usize>;
    fn length(&self) -> usize;
    fn increment_len(&mut self);
    fn decrement_len(&mut self);
    fn set_len(&mut self, len: usize);
    fn offset(&self) -> usize;
    fn set_offset(&mut self, offset: usize);
    fn load(&self) -> usize;
    fn set_load(&mut self, load: usize);
}

impl<T: InnerGetter> ListDataGetters for T {
    fn lists(&self) -> &[VecPy] {
        &self.inner().lists
    }
    fn lists_mut(&mut self) -> &mut Vec<VecPy> {
        &mut self.inner_mut().lists
    }
    fn maxes(&self) -> &[Py<PyAny>] {
        &self.inner().maxes
    }
    fn maxes_mut(&mut self) -> &mut VecPy {
        &mut self.inner_mut().maxes
    }
    fn idx(&self) -> &[usize] {
        &self.inner().idx
    }
    fn idx_mut(&mut self) -> &mut Vec<usize> {
        &mut self.inner_mut().idx
    }
    fn length(&self) -> usize {
        self.inner().len
    }
    fn set_len(&mut self, len: usize) {
        self.inner_mut().len = len;
    }
    fn increment_len(&mut self) {
        self.inner_mut().len += 1;
    }
    fn decrement_len(&mut self) {
        self.inner_mut().len -= 1;
    }
    fn offset(&self) -> usize {
        self.inner().offset
    }
    fn set_offset(&mut self, offset: usize) {
        self.inner_mut().offset = offset;
    }
    fn load(&self) -> usize {
        self.inner().load
    }
    fn set_load(&mut self, load: usize) {
        self.inner_mut().load = load;
    }
}

#[macro_export]
macro_rules! impl_inner_getter {
    ($name:ident) => {
        impl $crate::inner::InnerGetter for $name {
            fn inner(&self) -> &InnerData {
                &self.0
            }

            fn inner_mut(&mut self) -> &mut InnerData {
                &mut self.0
            }
        }
    };
}

fn get_slice<'a>(lists: &'a [VecPy], bounds: &Bounds) -> impl Iterator<Item = &'a Py<PyAny>> + 'a {
    lists[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(lists[bounds.min.pos + 1..bounds.max.pos].iter().flatten())
        .chain(lists[bounds.max.pos][0..bounds.max.idx].iter())
}
