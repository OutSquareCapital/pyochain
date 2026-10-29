use std::cmp::Ordering;

use crate::{
    Bounds, Loc, errors,
    lengths::Lengths,
    ops::Nb,
    traits::NestedVec,
    types::{SeqOrAny, VecPy},
};
use either::Either;
use itertools::Itertools;
use pyo3::{
    basic::CompareOp,
    exceptions::PyValueError,
    prelude::*,
    types::{PyList, PyNotImplemented, PySlice, slice::PySliceIndices},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use std_tools::prelude::*;
use tap::prelude::*;
pub struct InnerData {
    pub(super) values: Vec<VecPy>,
    pub maxes: VecPy,
    /// Equivalent to `idx` and `offset` attributes in the original `sortedcontainers` implementation.
    pub(super) lengths: Lengths,
    pub len: usize,
    pub(super) load: usize,
}

impl Default for InnerData {
    fn default() -> Self {
        Self {
            values: Vec::default(),
            maxes: Vec::default(),
            lengths: Lengths::default(),
            len: usize::default(),
            load: 1000,
        }
    }
}
impl InnerData {
    #[inline]
    pub fn clear(&mut self) {
        self.values.clear();
        self.maxes.clear();
        self.lengths.inner.clear();
        self.len = 0;
        self.lengths.offset = 0;
    }
    #[inline]
    #[must_use]
    pub fn collapse(&self, py: Python<'_>) -> VecPy {
        self.iter_owned(py).collect()
    }
    #[inline(always)]
    pub fn iter_owned(&self, py: Python<'_>) -> impl Iterator<Item = Py<PyAny>> {
        self.iter().map(move |x| x.clone_ref(py))
    }
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &Py<PyAny>> {
        self.values.iter().flatten()
    }
    pub fn as_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.iter().collect_bound::<PyList>(py)
    }
    pub fn loc(&mut self, loc: &Loc) -> usize {
        if loc.pos == 0 {
            loc.idx
        } else {
            if self.lengths.is_not_built() {
                self.lengths.build(&self.values);
            }
            self.lengths.loc_at(loc)
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
        if self.len == 0 {
            Err(errors::out_of_range())
        } else {
            let first_list = &self.values[0];
            let last_list = self.values.last().unwrap();
            let len_last = last_list.len().cast_signed();
            match index.conv::<Nb>() {
                Nb::Zero => first_list[0].clone_ref(py),
                Nb::MinusOne => last_list.last().unwrap().clone_ref(py),
                Nb::One | Nb::GTOne if index < first_list.len().cast_signed() => {
                    first_list[index.cast_unsigned()].clone_ref(py)
                }
                Nb::LTMinusOne if -len_last < index => {
                    last_list[(len_last + index).cast_unsigned()].clone_ref(py)
                }
                Nb::LTMinusOne | Nb::GTOne | Nb::One => {
                    let mut loc = Loc::default();
                    self.set_pos(index, &mut loc)?;
                    self.values.loc(&loc).clone_ref(py)
                }
            }
            .into_bound(py)
            .pipe(Ok)
        }
    }
    pub fn get_slice(&mut self, slice: &Bound<'_, PySlice>) -> PyResult<VecPy> {
        let py = slice.py();
        let PySliceIndices {
            start, stop, step, ..
        } = slice.indices(self.len.cast_signed())?;
        let stop_eq_len = stop.cmp(&self.len.cast_signed());
        let mut bounds = Bounds::default();
        match (start.cmp(&stop), step.conv::<Nb>()) {
            (_, Nb::Zero) => Err(PyValueError::new_err("slice step cannot be zero")),
            // Whole slice optimization: start to stop slices the whole sorted list.
            (Ordering::Less, Nb::One) if start == 0 && stop_eq_len.is_eq() => {
                self.collapse(py).pipe(Ok)
            }
            (Ordering::Less, Nb::One) => {
                self.set_pos(start, &mut bounds.min)?;
                let start_list = &self.values[bounds.min.pos];
                bounds.max.idx = bounds.min.idx + (stop - start).cast_unsigned();
                match (start_list.len().cmp(&bounds.max.idx), stop_eq_len) {
                    // Small slice optimization: start index and stop index are
                    // within the start list.
                    (Ordering::Equal | Ordering::Greater, _) => start_list
                        [bounds.min.idx..bounds.max.idx]
                        .iter()
                        .map(|x| x.clone_ref(py))
                        .collect::<Vec<_>>(),
                    (Ordering::Less, Ordering::Equal) => {
                        bounds.max.pos = self.values.len() - 1;
                        bounds.max.idx = self.values.loc_len(&bounds.max);
                        get_slice(&self.values, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                    }
                    (Ordering::Less, Ordering::Greater | Ordering::Less) => {
                        self.set_pos(stop, &mut bounds.max)?;
                        get_slice(&self.values, &bounds)
                            .map(|x| x.clone_ref(py))
                            .collect::<Vec<_>>()
                    }
                }
                .pipe(Ok)
            }
            (Ordering::Greater, Nb::MinusOne) => {
                let mut result = self.get_slice(&PySlice::new(py, stop + 1, start + 1, 1))?;
                result.reverse();
                Ok(result)
            }
            // Negative step with nothing to iterate (mirrors Python's `range`,
            // which is empty when `start <= stop` for a negative step).
            (Ordering::Less | Ordering::Equal, Nb::MinusOne | Nb::LTMinusOne) => Ok(Vec::new()),
            (Ordering::Greater, Nb::LTMinusOne) => {
                std::iter::successors(Some(start), move |&i| (i + step > stop).then_some(i + step))
                    .map(|i| self.get_item(py, i))
                    .map_ok(Bound::unbind)
                    .collect()
            }
            // Return a list because a negative step could reverse the order
            // of the items and this could be the desired behavior.
            (_, Nb::One | Nb::GTOne) => (start..stop)
                .step_by(step.cast_unsigned())
                .map(|i| self.get_item(py, i))
                .map_ok(Bound::unbind)
                .collect(),
        }
    }

    pub(super) fn remove_pos(&mut self, loc: &Loc) {
        self.values.remove(loc.pos);
        self.maxes.remove(loc.pos);
        self.lengths.inner.clear();
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
        self.values.insert(pos + 1, half);
        self.lengths.inner.clear();
    }

    pub(super) fn set_pos(&mut self, mut idx: isize, loc: &mut Loc) -> PyResult<()> {
        if idx < 0 {
            if idx >= -self.values.last().unwrap().len().cast_signed() {
                loc.pos = self.values.len() - 1;
                loc.idx = (self.values.last().unwrap().len().cast_signed() + idx).cast_unsigned();
                return Ok(());
            }

            idx += self.len.cast_signed();

            if idx < 0 {
                return Err(errors::out_of_range());
            }
        } else if idx >= self.len.cast_signed() {
            return Err(errors::out_of_range());
        }

        if idx < self.values[0].len().cast_signed() {
            loc.pos = 0;
            loc.idx = idx.cast_unsigned();
            Ok(())
        } else {
            if self.lengths.is_not_built() {
                self.lengths.build(&self.values);
            }
            self.lengths.pos_at(idx, loc);
            Ok(())
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
