use crate::{
    bounds::Loc,
    errors,
    lengths::Lengths,
    ops,
    traits::NestedVec,
    types::{SeqOrAny, VecPy},
};
use derive_more::{Deref, From};
use either::Either;
use pyo3::{
    basic::CompareOp,
    prelude::*,
    types::{PyList, PyNotImplemented},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut},
};
use std_tools::prelude::*;
#[derive(Clone, Copy, Debug, From, Deref)]
pub(super) struct Load(usize);
impl Default for Load {
    fn default() -> Self {
        Self(1000)
    }
}
impl Load {
    /// This was present in the original `sortedcontainers` tests. Must check if really needed.
    pub const fn test_limit(self) -> bool {
        self.0 >= 4
    }
    #[inline(always)]
    pub const fn higher_limit(self) -> usize {
        self.0 * 2
    }
    #[inline(always)]
    pub const fn lower_limit(self) -> usize {
        self.0 / 2
    }
}
#[derive(Default)]
pub struct InnerData {
    pub(super) values: Vec<VecPy>,
    pub(super) maxes: VecPy,
    /// Equivalent to `idx` and `offset` attributes in the original `sortedcontainers` implementation.
    pub(super) lengths: Lengths,
    pub(super) len: usize,
    pub(super) load: Load,
}
impl InnerData {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub(super) fn clear(&mut self) {
        self.values.clear();
        self.maxes.clear();
        self.lengths.inner.clear();
        self.len = 0;
        self.lengths.offset = 0;
    }
    #[inline]
    #[must_use]
    pub(super) fn collapse(&self, py: Python<'_>) -> VecPy {
        self.iter_owned(py).collect()
    }
    #[inline(always)]
    pub(super) fn iter_owned(&self, py: Python<'_>) -> impl DoubleEndedIterator<Item = Py<PyAny>> {
        self.iter().map(move |x| x.clone_ref(py))
    }
    #[inline(always)]
    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &Py<PyAny>> {
        self.values.iter().flatten()
    }
    pub(super) fn as_pylist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.iter().collect_bound::<PyList>(py)
    }
    pub(super) fn loc(&mut self, loc: &Loc) -> usize {
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
    pub(super) fn repeat(&self, py: Python<'_>, num: usize) -> VecPy {
        let values = self.collapse(py);
        (0..num)
            .flat_map(|_| values.iter())
            .map(|x| x.clone_ref(py))
            .collect()
    }
    #[inline]
    pub(super) fn index_to_loc(&mut self, index: isize) -> PyResult<Loc> {
        match ops::Locate::new(self.len, &self.values, index) {
            ops::Locate::Empty => Err(errors::out_of_range()),
            ops::Locate::Last => {
                let pos = self.values.len() - 1;
                let idx = self.values[pos].len() - 1_usize;
                Ok(Loc::new(pos, idx))
            }
            ops::Locate::First => Ok(Loc::with_idx(0)),
            ops::Locate::InFirstList(i) => Ok(Loc::with_idx(i)),
            ops::Locate::InLastList(i) => {
                let pos = self.values.len() - 1;
                let idx = (self.values.last().unwrap().len().cast_signed() + i).cast_unsigned();
                Ok(Loc::new(pos, idx))
            }
            ops::Locate::InOtherList(i) => self.set_pos_usize(i),
            ops::Locate::InOtherListNeg(i) => self.set_pos(i),
        }
    }
    pub(super) fn get_item<'py>(
        &mut self,
        py: Python<'py>,
        index: isize,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.index_to_loc(index)
            .map(|loc| self.values.loc(&loc).clone_ref(py).into_bound(py))
    }

    pub(super) fn remove_pos(&mut self, loc: &Loc) {
        self.values.remove(loc.pos);
        self.maxes.remove(loc.pos);
        self.lengths.inner.clear();
    }

    pub(super) fn set_pos(&mut self, mut index: isize) -> PyResult<Loc> {
        if index.is_negative() {
            let last_len = self.values.last().unwrap().len().cast_signed();
            if index >= -last_len {
                let pos = self.values.len() - 1;
                let idx = (last_len + index).cast_unsigned();
                return Ok(Loc::new(pos, idx));
            }
            index += self.len.cast_signed();
            if index.is_negative() {
                return Err(errors::out_of_range());
            }
        }
        self.set_pos_usize(index.cast_unsigned())
    }
    pub(super) fn set_pos_usize(&mut self, index: usize) -> PyResult<Loc> {
        if index >= self.len {
            Err(errors::out_of_range())
        } else if index < self.values[0].len() {
            Ok(Loc::with_idx(index))
        } else {
            if self.lengths.is_not_built() {
                self.lengths.build(&self.values);
            }
            Ok(self.lengths.pos_at(index))
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
