use crate::{
    bounds::Loc,
    errors,
    lengths::Lengths,
    load::Load,
    ops,
    sublist::Chunks,
    types::{SeqOrAny, VecPy},
};
use derive_more::{Deref, DerefMut, From, IntoIterator};
use pyo3::{BoundObject, basic::CompareOp, prelude::*, types::PyNotImplemented};
use pyo3_ext::{prelude::*, types::PyCmpOut};
use std_tools::prelude::*;
use tap::prelude::*;
#[derive(Deref, DerefMut, From, Debug, Default, IntoIterator)]
pub struct Maxes(VecPy);
impl Maxes {
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        self.iter()
            .map(|x| x.clone_ref(py))
            .collect::<VecPy>()
            .into()
    }
}
#[derive(Default)]
pub struct InnerList {
    pub(super) values: Chunks,
    pub(super) maxes: Maxes,
    /// Equivalent to `idx` and `offset` attributes in the original `sortedcontainers` implementation.
    pub(super) lengths: Lengths,
    pub(super) len: usize,
    pub(super) load: Load,
}
impl InnerList {
    /// Duplicates the sublists, the maxes, the length and the load.\
    /// The lengths index is only a cache, and is rebuilt on demand.
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self {
            values: self.values.clone_ref(py),
            maxes: self.maxes.clone_ref(py),
            lengths: Lengths::default(),
            len: self.len,
            load: self.load,
        }
    }
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
    pub(super) fn index_to_loc(&mut self, index: isize) -> PyResult<Loc> {
        match ops::Locate::new(self, index) {
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
                let idx = (self.values[pos].len().cast_signed() + i).cast_unsigned();
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

    pub(super) fn set_pos(&mut self, mut index: isize) -> PyResult<Loc> {
        if index.is_negative() {
            let pos = self.values.len() - 1;
            let last_len = self.values[pos].len().cast_signed();
            if index >= -last_len {
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
        self.inner_cmp(other, CompareOp::Eq)
    }
    pub fn ne<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        self.inner_cmp(other, CompareOp::Ne)
    }
    pub fn lt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        self.inner_cmp(other, CompareOp::Lt)
    }
    pub fn gt<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        self.inner_cmp(other, CompareOp::Gt)
    }
    pub fn le<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        self.inner_cmp(other, CompareOp::Le)
    }
    pub fn ge<'py>(&self, other: SeqOrAny<'py>) -> PyCmpOut<'py, bool> {
        self.inner_cmp(other, CompareOp::Ge)
    }
    #[inline]
    fn inner_cmp<'py>(&self, other: SeqOrAny<'py>, op: CompareOp) -> PyCmpOut<'py, bool> {
        let py = either::for_both!(&other, x => x.py());
        other
            .map_left(|seq| {
                let mut it = self.values.iter().map(|a| a.bind(py)).zip(seq.iter_py());
                match (op, self.len == seq.len()?) {
                    (CompareOp::Eq, true) => it.try_all(|(a, b)| a.eq(b?)),
                    (CompareOp::Eq, false) => Ok(false),
                    (CompareOp::Ne, true) => it.try_any(|(a, b)| a.ne(b?)),
                    (CompareOp::Ne, false) => Ok(true),
                    (op, _) => it
                        .try_find_map(|(a, b)| {
                            let b = b?;
                            if a.ne(&b)? {
                                a.rich_compare_bool(&b, op).map(Some)
                            } else {
                                Ok(None)
                            }
                        })?
                        .map_or_else(|| Ok(op.as_fn::<usize>()(&self.len, &seq.len()?)), Ok),
                }
            })
            .map_right(|_| PyNotImplemented::get(py).into_bound().pipe(Ok))
            .factor_err()
    }
}
