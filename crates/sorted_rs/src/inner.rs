use crate::{
    bisect::bisect, bounds::Loc, chunks::ChunksMethods, delete, errors, item::Item,
    lengths::Lengths, load::Load, maxes::Maxes, ops, traits::BisectRes, types::SeqOrAny,
};
use derive_new::new;
use pyo3::{BoundObject, basic::CompareOp, prelude::*, types::PyNotImplemented};
use pyo3_ext::{prelude::*, types::PyCmpOut};
use std_tools::prelude::*;
use tap::prelude::*;
#[derive(Default, new)]
pub struct InnerList<C: ChunksMethods> {
    pub(super) chunks: C,
    pub(super) maxes: Maxes,
    /// Equivalent to `idx` and `offset` attributes in the original `sortedcontainers` implementation.
    #[new(default)]
    pub(super) lengths: Lengths,
    pub(super) len: usize,
    pub(super) load: Load,
}
impl<C: ChunksMethods> InnerList<C> {
    /// Duplicates the sublists, the maxes, the length and the load.\
    /// The lengths index is only a cache, and is rebuilt on demand.
    pub(super) fn clone_ref(&self, py: Python<'_>) -> Self {
        Self::new(
            self.chunks.clone_ref(py),
            self.maxes.clone_ref(py),
            self.len,
            self.load,
        )
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
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.maxes.clear();
        self.lengths.inner.clear();
        self.len = 0;
        self.lengths.offset = 0;
    }
    pub fn delete(&mut self, py: Python<'_>, loc: Loc) {
        self.len -= 1;
        delete::Strategy::new(self, loc.pos).dispatch(py, self, loc);
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
    pub fn pop<'py>(&mut self, py: Python<'py>, index: isize) -> PyResult<Bound<'py, PyAny>> {
        let loc = self.index_to_loc(index)?;
        let val = self.chunks.values().get_at(loc).clone_ref(py);
        self.delete(py, loc);
        Ok(val.into_bound(py))
    }
    pub fn bisect_by_key<const LEFT: bool>(&mut self, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        match self.try_bisect::<LEFT>(value)? {
            ops::Bisect::Empty => Ok(0),
            ops::Bisect::MaxesLen(_) => Ok(self.len()),
            ops::Bisect::AtLoc(loc) => Ok(loc.into_index(self)),
        }
    }
    pub(crate) fn del_item(&mut self, py: Python<'_>, index: isize) -> PyResult<()> {
        if self.is_empty() {
            Err(errors::out_of_range())
        } else {
            self.set_pos(index).map(|loc| self.delete(py, loc))
        }
    }
    pub(crate) fn expand(&mut self, py: Python<'_>, pos: usize) {
        match ops::Expand::new(self, pos) {
            ops::Expand::MustSplit => {
                let load = *self.load;
                let last_max = self.chunks.split(py, pos, load);
                self.maxes.insert(pos + 1, last_max);
                self.refresh_max(py, pos);
                self.lengths.inner.clear();
            }
            ops::Expand::IdxBuilt => self.lengths.expand_at(pos),
            ops::Expand::NoOp => (),
        }
    }
    pub(super) fn append_item(&mut self, py: Python<'_>, item: C::Item) -> PyResult<()> {
        match self.try_bisect_right(item.key())? {
            ops::Bisect::Empty => {
                self.maxes.push(item.key().clone().unbind());
                self.chunks.push_new_chunk(item);
            }
            ops::Bisect::MaxesLen(pos) => {
                self.maxes[pos] = item.key().clone().unbind();
                self.chunks.push_in_chunk(pos, item);
                self.expand(py, pos);
            }
            ops::Bisect::AtLoc(loc) => self.insert_in_chunk(py, loc, item),
        }
        self.len += 1;
        Ok(())
    }
    pub(super) fn insert_in_chunk(&mut self, py: Python<'_>, loc: Loc, item: C::Item) {
        self.chunks.insert_in_chunk(loc, item);
        self.expand(py, loc.pos);
    }
    pub(super) fn remove_in_chunk(&mut self, py: Python<'_>, loc: Loc) {
        self.chunks.remove_in_chunk(loc);
        self.refresh_max(py, loc.pos);
    }
    #[inline]
    pub(super) fn refresh_max(&mut self, py: Python<'_>, pos: usize) {
        self.maxes[pos] = self.chunks.keys()[pos].last().clone_ref(py);
    }
    #[inline]
    fn index_to_loc(&mut self, index: isize) -> PyResult<Loc> {
        match ops::Locate::new(self, index) {
            ops::Locate::Empty => Err(errors::out_of_range()),
            ops::Locate::Last => {
                let pos = self.chunks.len() - 1;
                let idx = self.chunks.keys()[pos].len() - 1_usize;
                Ok(Loc::new(pos, idx))
            }
            ops::Locate::First => Ok(Loc::with_idx(0)),
            ops::Locate::InFirstList(i) => Ok(Loc::with_idx(i)),
            ops::Locate::InLastList(i) => {
                let pos = self.chunks.len() - 1;
                let idx = (self.chunks.values()[pos].len().cast_signed() + i).cast_unsigned();
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
        self.index_to_loc(index).map(|loc| {
            self.chunks
                .values()
                .get_at(loc)
                .clone_ref(py)
                .into_bound(py)
        })
    }

    pub(super) fn set_pos(&mut self, mut index: isize) -> PyResult<Loc> {
        if index.is_negative() {
            let pos = self.chunks.values().len() - 1;
            let last_len = self.chunks.values()[pos].len().cast_signed();
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
        } else if index < self.chunks.values()[0].len() {
            Ok(Loc::with_idx(index))
        } else {
            self.may_build();
            Ok(self.lengths.pos_at(index))
        }
    }
    pub(super) fn may_build(&mut self) {
        if self.lengths.is_not_built() {
            self.lengths.build(self.chunks.values());
        }
    }

    /// Location of `key`, the item following the one at `Loc` in a sorted batch.\
    /// Only compares from `Loc`, since `key` can't go before it.
    /// Stays valid if the previous item was already inserted at `Loc`, and its sublist split.\
    /// Uses `bisect_right` semantics like `append`: a key equal to a max goes into the next sublist,
    /// and a key greater than every max goes into the last one.
    pub(super) fn next_sorted(&self, loc: Loc, key: &Bound<'_, PyAny>) -> PyResult<Loc> {
        let last = self.maxes.len() - 1;
        let pos = if loc.pos == last
            || key.rich_compare_bool(self.maxes[loc.pos].bind(key.py()), CompareOp::Lt)?
        {
            loc.pos
        } else {
            bisect::<false>(&self.maxes[loc.pos + 1..last], key)? + loc.pos + 1
        };
        let low = if pos == loc.pos { loc.idx } else { 0 };
        let idx = bisect::<false>(&self.chunks.keys()[pos][low..], key)?;
        Ok(Loc::new(pos, low + idx))
    }
    pub(super) fn push_chunk(&mut self, py: Python<'_>, chunk: C::IntoChunk) {
        let max = self.chunks.push_chunk(py, chunk);
        self.maxes.push(max);
    }
    #[inline(always)]
    pub(super) fn try_bisect<const LEFT: bool>(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        ops::Bisect::new::<LEFT, C>(self, value)
    }
    #[inline(always)]
    pub(super) fn try_bisect_left(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        self.try_bisect::<true>(value)
    }
    #[inline(always)]
    pub(super) fn try_bisect_right(&self, value: &Bound<'_, PyAny>) -> BisectRes {
        self.try_bisect::<false>(value)
    }
    #[inline]
    fn inner_cmp<'py>(&self, other: SeqOrAny<'py>, op: CompareOp) -> PyCmpOut<'py, bool> {
        let py = either::for_both!(&other, x => x.py());
        other
            .map_left(|seq| {
                let mut it = self
                    .chunks
                    .values()
                    .iter()
                    .map(|a| a.bind(py))
                    .zip(seq.iter_py());
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
