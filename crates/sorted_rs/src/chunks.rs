use std::vec::Drain;

use derive_more::{Constructor, Deref, DerefMut, From, IntoIterator};
use pyo3::prelude::*;
use std_tools::collections::WithCapacity;

use crate::{
    bounds::Loc,
    delete,
    item::{Item, ItemRef},
    types::{Pair, VecPy},
};
pub trait ChunksMethods {
    type Item: Item;
    type IntoChunk: WithCapacity<Self::Item>;
    /// Returns a reference to the "keys" of the collection.\
    /// For collections sorted by *values*, this is equivalent to `InnerData::values`.\
    /// For collections sorted by *key*, this is equivalent to `KeysListsData::1`.
    fn keys(&self) -> &Chunks;
    fn keys_mut(&mut self) -> &mut Chunks;
    fn clone_ref(&self, py: Python<'_>) -> Self;
    fn values(&self) -> &Chunks;
    /// Moves the last `run.len()` items of the chunk into the sublist of the run,
    /// each item at the matching location of `run`.
    fn drain_from(&mut self, chunk: &mut Self::IntoChunk, idxs: &[Loc]);
    fn values_mut(&mut self) -> &mut Chunks;
    fn remove_in_chunk(&mut self, loc: Loc);
    fn remove_chunk(&mut self, pos: usize);
    fn split(&mut self, py: Python<'_>, pos: usize, load: usize) -> Py<PyAny>;
    fn merge(&mut self, loc: Loc, merge: delete::Merge) -> usize;
    fn loc_item(&self, loc: Loc) -> impl ItemRef<Owned = Self::Item>;
    /// Inserts `item` into the list at `loc`.
    fn insert_in_chunk(&mut self, loc: Loc, item: Self::Item);
    fn clear(&mut self);
    fn push_chunk(&mut self, py: Python<'_>, chunk: Self::IntoChunk) -> Py<PyAny>;
    fn push_in_chunk(&mut self, pos: usize, item: Self::Item);
    fn push_new_chunk(&mut self, item: Self::Item);
    fn len(&self) -> usize {
        self.keys().len()
    }
    fn is_empty(&self) -> bool {
        self.keys().is_empty()
    }
}
#[derive(Constructor, Default)]
pub struct ChunksWithKeys {
    keys: Chunks,
    values: Chunks,
}
#[derive(Deref, DerefMut, From, Debug, Default, IntoIterator)]
pub struct Chunks(Vec<Chunk>);
impl ChunksMethods for ChunksWithKeys {
    type Item = Pair<Py<PyAny>>;
    type IntoChunk = ChunkKey;
    fn keys(&self) -> &Chunks {
        &self.keys
    }
    fn keys_mut(&mut self) -> &mut Chunks {
        &mut self.keys
    }
    fn values(&self) -> &Chunks {
        &self.values
    }
    fn values_mut(&mut self) -> &mut Chunks {
        &mut self.values
    }
    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }
    fn clone_ref(&self, py: Python<'_>) -> Self {
        Self::new(self.keys.clone_ref(py), self.values.clone_ref(py))
    }
    fn loc_item(&self, loc: Loc) -> impl ItemRef<Owned = Self::Item> {
        (self.keys().get_at(loc), self.values().get_at(loc))
    }
    fn split(&mut self, py: Python<'_>, pos: usize, load: usize) -> Py<PyAny> {
        let max = self.keys.split(py, pos, load);
        let half = self.values.0[pos].split_off(load);
        self.values.0.insert(pos + 1, half);
        max
    }
    fn merge(&mut self, loc: Loc, merge: delete::Merge) -> usize {
        self.keys.merge(loc, merge);
        self.values.merge(loc, merge)
    }
    fn remove_chunk(&mut self, pos: usize) {
        self.keys.remove_chunk(pos);
        self.values.remove_chunk(pos);
    }
    fn insert_in_chunk(&mut self, loc: Loc, (key, value): Self::Item) {
        self.keys_mut().insert_in_chunk(loc, key);
        self.values_mut().insert_in_chunk(loc, value);
    }
    fn remove_in_chunk(&mut self, loc: Loc) {
        self.keys_mut().remove_in_chunk(loc);
        self.values_mut().remove_in_chunk(loc);
    }
    fn drain_from(&mut self, chunk: &mut Self::IntoChunk, idxs: &[Loc]) {
        self.keys.drain_from(&mut chunk.keys, idxs);
        let start = chunk.values.len() - idxs.len();
        self.values_mut()[idxs[0].pos].insert_many(idxs, chunk.values.drain(start..));
    }
    fn push_chunk(&mut self, py: Python<'_>, chunk: Self::IntoChunk) -> Py<PyAny> {
        let max = self.keys.push_chunk(py, chunk.keys);
        self.values.0.push(chunk.values.into());
        max
    }
    fn push_in_chunk(&mut self, pos: usize, (key, value): Self::Item) {
        self.keys.push_in_chunk(pos, key);
        self.values.push_in_chunk(pos, value);
    }
    fn push_new_chunk(&mut self, (key, value): Self::Item) {
        self.keys.push_new_chunk(key);
        self.values.push_new_chunk(value);
    }
}
impl ChunksMethods for Chunks {
    type Item = Py<PyAny>;
    type IntoChunk = VecPy;
    fn values(&self) -> &Chunks {
        self
    }
    fn values_mut(&mut self) -> &mut Chunks {
        self
    }
    fn keys(&self) -> &Chunks {
        self
    }
    fn keys_mut(&mut self) -> &mut Chunks {
        self
    }
    fn clear(&mut self) {
        self.0.clear();
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn clone_ref(&self, py: Python<'_>) -> Self {
        self.iter_chunks()
            .map(|chunk| chunk.clone_ref(py))
            .collect::<Vec<_>>()
            .into()
    }
    fn remove_chunk(&mut self, pos: usize) {
        self.0.remove(pos);
    }
    fn remove_in_chunk(&mut self, loc: Loc) {
        self[loc.pos].remove(loc.idx);
    }
    fn insert_in_chunk(&mut self, loc: Loc, item: Self::Item) {
        self[loc.pos].insert(loc.idx, item);
    }
    fn loc_item(&self, loc: Loc) -> impl ItemRef<Owned = Self::Item> {
        self.get_at(loc)
    }
    fn split(&mut self, py: Python<'_>, pos: usize, load: usize) -> Py<PyAny> {
        let half = self.0[pos].split_off(load);
        let max = half.last().clone_ref(py);
        self.0.insert(pos + 1, half);
        max
    }
    fn merge(&mut self, loc: Loc, merge: delete::Merge) -> usize {
        merge.remove(self, loc)
    }
    fn drain_from(&mut self, chunk: &mut Self::IntoChunk, idxs: &[Loc]) {
        let start = chunk.len() - idxs.len();
        self.keys_mut()[idxs[0].pos].insert_many(idxs, chunk.drain(start..));
    }
    fn push_chunk(&mut self, py: Python<'_>, chunk: Self::IntoChunk) -> Py<PyAny> {
        let chunk = Chunk::from(chunk);
        let max = chunk.last().clone_ref(py);
        self.0.push(chunk);
        max
    }
    fn push_in_chunk(&mut self, pos: usize, item: Self::Item) {
        self.0[pos].push(item);
    }
    fn push_new_chunk(&mut self, item: Self::Item) {
        self.0.push(Chunk::new(item));
    }
}
impl Chunks {
    #[inline(always)]
    pub(super) fn iter_owned(&self, py: Python<'_>) -> impl DoubleEndedIterator<Item = Py<PyAny>> {
        self.iter().map(move |x| x.clone_ref(py))
    }
    #[inline(always)]
    pub(super) fn iter(&self) -> impl DoubleEndedIterator<Item = &Py<PyAny>> + Clone {
        self.0.iter().flatten()
    }
    pub(super) fn iter_chunks(&self) -> impl DoubleEndedIterator<Item = &Chunk> {
        self.0.iter()
    }
    pub(super) fn get_at(&self, loc: Loc) -> &Py<PyAny> {
        &self[loc.pos][loc.idx]
    }
    pub(super) fn last_in_chunk(&self, loc: &Loc) -> &Py<PyAny> {
        self[loc.pos].last()
    }
    pub(super) fn len_of_chunk(&self, loc: &Loc) -> usize {
        self[loc.pos].len()
    }
}

#[derive(Constructor, From)]
pub struct ChunkKey {
    keys: VecPy,
    values: VecPy,
}
impl FromIterator<Pair<Py<PyAny>>> for ChunkKey {
    fn from_iter<I: IntoIterator<Item = Pair<Py<PyAny>>>>(iter: I) -> Self {
        iter.into_iter().collect::<Pair<Vec<_>>>().into()
    }
}
impl WithCapacity<Pair<Py<PyAny>>> for ChunkKey {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }
    fn push(&mut self, (key, value): Pair<Py<PyAny>>) {
        self.keys.push(key);
        self.values.push(value);
    }
}
/// A sublist of a sorted list. Never empty.\
/// Only gives access to its items as a slice, so its length can only change through the methods below,
/// which never leave it empty.
#[derive(Debug, Deref, IntoIterator)]
#[deref(forward)]
#[into_iterator(ref)]
pub struct Chunk(VecPy);
impl From<VecPy> for Chunk {
    fn from(vec: VecPy) -> Self {
        debug_assert!(!vec.is_empty());
        Self(vec)
    }
}
impl Chunk {
    pub fn new(first: Py<PyAny>) -> Self {
        Self(vec![first])
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, Py<PyAny>> {
        self.0.iter()
    }
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self(self.0.iter().map(|x| x.clone_ref(py)).collect())
    }
    pub fn last(&self) -> &Py<PyAny> {
        // SAFETY: the sublist is never empty, so `last` always returns `Some`.
        unsafe { self.0.last().unwrap_unchecked() }
    }
    pub fn push(&mut self, value: Py<PyAny>) {
        self.0.push(value);
    }
    pub fn insert(&mut self, idx: usize, value: Py<PyAny>) {
        self.0.insert(idx, value);
    }
    /// Inserts every item of `items` before the element that is at the `idx` of the matching location of `locs`
    /// in the original `Vec`.\
    /// Items sharing the same index keep their relative order.
    ///
    /// Equivalent to calling `insert` for each item in reverse order, but each element is moved at most once,
    /// in a single backward pass, instead of shifting the whole tail on every insertion.\
    /// Takes a [`Drain`] so the caller can reuse the buffer the items come from.
    ///
    /// # Panics
    ///
    /// Panics if `idxs` and `items` don't have the same length,
    /// or if `idxs` isn't sorted or one is greater than the length.
    pub fn insert_many(&mut self, idxs: &[Loc], items: Drain<'_, Py<PyAny>>) {
        let old_len = self.len();
        // `Drain` is a std `ExactSizeIterator`: the unsafe code below can trust its length.
        let new_len = old_len + items.len();
        debug_assert_eq!(idxs.len(), items.len(), "one location is expected per item");
        debug_assert!(
            idxs.is_sorted_by_key(|loc| loc.idx)
                && idxs.last().is_none_or(|last| last.idx <= old_len),
            "indices must be sorted and at most the length"
        );
        self.0.reserve(items.len());
        let base = self.0.as_mut_ptr();
        let mut src_end = old_len;
        let mut dst_end = new_len;
        items.zip(idxs).rev().for_each(|(item, &Loc { idx, .. })| {
            let count = src_end - idx;
            dst_end -= count + 1;
            // SAFETY: the capacity holds `new_len` elements.
            // `[idx, src_end)` are initialized elements not moved yet, and since `dst_end >= src_end` is kept
            // by the sorted indices, the destination `[dst_end + 1, dst_end + 1 + count)` only overlaps
            // elements that were already moved or never initialized. `ptr::copy` handles the overlap.
            unsafe {
                std::ptr::copy(base.add(idx), base.add(dst_end + 1), count);
                std::ptr::write(base.add(dst_end), item);
            }
            src_end = idx;
        });
        // SAFETY: every slot of `[0, new_len)` is now initialized exactly once:
        // `[0, src_end)` never moved, and `dst_end == src_end` once all items are written.
        unsafe { self.0.set_len(new_len) };
    }
    /// The sublist must keep at least one item.
    pub fn remove(&mut self, idx: usize) {
        debug_assert!(self.0.len() > 1);
        self.0.remove(idx);
    }
    /// Both halves must keep at least one item.
    pub fn split_off(&mut self, at: usize) -> Self {
        debug_assert!(0 < at && at < self.0.len());
        Self(self.0.split_off(at))
    }
    pub fn append(&mut self, other: &mut VecPy) {
        self.0.append(other);
    }
    pub fn into_vec(self) -> VecPy {
        self.0
    }
}
