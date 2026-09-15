use std::cmp::Ordering;

use crate::{
    Bounds, Loc, errors,
    indexing::Nb,
    traits::NestedVec,
    types::{SeqOrAny, VecPy},
};
use either::Either;
use pyo3::{
    basic::CompareOp,
    prelude::*,
    types::{PyList, PyNotImplemented, PySlice, PySliceIndices},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use std_tools::prelude::*;
use tap::{Conv, Pipe};
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

    pub fn as_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.iter().collect_bound::<PyList>(py)
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
        let first_list = &self.lists[0];
        let last_list = self.lists.last().unwrap();
        let len_last = last_list.len().cast_signed();
        match (index.conv::<Nb>(), self.len.cmp(&0)) {
            (Nb::Zero, Ordering::Greater | Ordering::Less) => {
                first_list[0].clone_ref(py).into_bound(py).pipe(Ok)
            }
            (Nb::NegOne, Ordering::Greater | Ordering::Less) => last_list
                .last()
                .unwrap()
                .clone_ref(py)
                .into_bound(py)
                .pipe(Ok),
            (_, Ordering::Equal) => Err(errors::out_of_range()),
            (Nb::One | Nb::Pos, Ordering::Greater | Ordering::Less)
                if index < first_list.len().cast_signed() =>
            {
                first_list[index.cast_unsigned()]
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
            (Nb::Neg, Ordering::Greater | Ordering::Less) if -len_last < index => last_list
                [(len_last + index).cast_unsigned()]
            .clone_ref(py)
            .into_bound(py)
            .pipe(Ok),
            _ => {
                let mut bounds = Bounds::default();
                self.set_pos(index, &mut bounds.min)?;
                self.lists
                    .loc(&bounds.min)
                    .clone_ref(py)
                    .into_bound(py)
                    .pipe(Ok)
            }
        }
    }
    pub fn get_slice(&mut self, slice: &Bound<'_, PySlice>) -> PyResult<VecPy> {
        let py = slice.py();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.len.cast_signed())?;
        let stop_eq_len = stop.cmp(&self.len.cast_signed());
        let mut bounds = Bounds::default();
        match (step.conv::<Nb>(), start.cmp(&stop)) {
            // Whole slice optimization: start to stop slices the whole sorted list.
            (Nb::One, Ordering::Less) if start == 0 && stop_eq_len.is_eq() => {
                self.collapse(py).pipe(Ok)
            }
            (Nb::One, Ordering::Less) => {
                self.set_pos(start, &mut bounds.min)?;
                let start_list = &self.lists[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (stop - start).cast_unsigned();
                match (start_list.len().cmp(&bounds.max.idx), stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (Ordering::Equal | Ordering::Greater, _) => start_list
                        [bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>()
                        .pipe(Ok),
                    (Ordering::Less, Ordering::Equal) => {
                        bounds.max.pos = self.lists.len() - 1;
                        bounds.max.idx = self.lists.loc_len(&bounds.max);
                        get_slice(&self.lists, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                    (Ordering::Less, Ordering::Greater | Ordering::Less) => {
                        self.set_pos(stop, &mut bounds.max)?;
                        get_slice(&self.lists, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                            .pipe(Ok)
                    }
                }
            }
            (Nb::NegOne, Ordering::Greater) => {
                let mut result = self.get_slice(&PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            (Nb::One | Nb::Pos, _) => (start..stop)
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
                return Err(errors::out_of_range());
            }
        } else if idx >= self.len.cast_signed() {
            return Err(errors::out_of_range());
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
        inner_cmp(self, other, CompareOp::Eq)
    }
    pub fn ne<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        inner_cmp(self, other, CompareOp::Ne)
    }
    pub fn lt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        inner_cmp(self, other, CompareOp::Lt)
    }
    pub fn gt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        inner_cmp(self, other, CompareOp::Gt)
    }
    pub fn le<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        inner_cmp(self, other, CompareOp::Le)
    }
    pub fn ge<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        inner_cmp(self, other, CompareOp::Ge)
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

fn get_slice<'a>(lists: &'a [VecPy], bounds: &Bounds) -> impl Iterator<Item = &'a Py<PyAny>> + 'a {
    lists[bounds.min.pos][bounds.min.idx..]
        .iter()
        .chain(lists[bounds.min.pos + 1..bounds.max.pos].iter().flatten())
        .chain(lists[bounds.max.pos][0..bounds.max.idx].iter())
}

#[inline]
fn inner_cmp<'py>(data: &InnerData, other: SeqOrAny<'py>, op: CompareOp) -> PyCmpOut<'py, bool> {
    let py = other.py();
    let it = data.iter().map(|a| a.bind(py));
    match (other, op) {
        (Either::Left(seq), CompareOp::Eq) => {
            if data.len == seq.len()? {
                it.zip(seq.iter_py())
                    .try_all(|(a, b)| a.eq(b?))
                    .map(Either::Left)
            } else {
                Ok(Either::Left(false))
            }
        }
        (Either::Left(seq), CompareOp::Ne) => {
            if data.len == seq.len()? {
                it.zip(seq.iter_py())
                    .try_any(|(a, b)| a.ne(b?))
                    .map(Either::Left)
            } else {
                Ok(Either::Left(true))
            }
        }
        (Either::Left(seq), op) => it
            .zip(seq.iter_py())
            .try_find_map(|(a, b)| {
                let b = b?;
                if a.ne(&b)? {
                    a.rich_compare_bool(&b, op).map(Some)
                } else {
                    Ok(None)
                }
            })?
            .map_or_else(|| Ok(op.as_fn::<usize>()(&data.len, &seq.len()?)), Ok)
            .map(Either::Left),
        (Either::Right(any), _) => PyNotImplemented::from_cmp(any.py()),
    }
}
