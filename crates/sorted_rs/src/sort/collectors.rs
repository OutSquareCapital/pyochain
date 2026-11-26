use crate::{
    bounds::Loc, chunks::ChunksMethods, inner::InnerList, item::Item, sort::sorted::SortedVec,
};

use derive_more::{Deref, From};
use derive_new::new;
use pyo3::prelude::*;
use std::iter::RepeatN;
use std_tools::prelude::TryIterator;
use tap::Pipe;

pub(crate) trait IntoChunked<T> {
    fn into_chunked(self) -> Chunked<Self>
    where
        Self: Sized + ExactSizeIterator<Item = T>,
    {
        Chunked(self)
    }
}
impl<T, I: ExactSizeIterator<Item = T>> IntoChunked<T> for I {}

pub(crate) struct Chunked<I>(I);
impl<I: ExactSizeIterator> Chunked<I> {
    pub fn extend_into<C: ChunksMethods<Item = I::Item>>(
        mut self,
        py: Python<'_>,
        list: &mut InnerList<C>,
    ) {
        self.chunk_lens(list)
            .map(|len| self.0.by_ref().take(len).collect::<C::IntoChunk>())
            .for_each(|chunk| list.push_chunk(py, chunk));
    }
    fn chunk_lens(&self, list: &mut InnerList<impl ChunksMethods>) -> RepeatN<usize> {
        debug_assert!(list.is_empty());
        let length = self.0.len();
        let load = *list.load;
        list.len = length;
        std::iter::repeat_n(load, length.div_ceil(load))
    }
}
impl<T, I: ExactSizeIterator<Item = PyResult<T>>> Chunked<I> {
    pub fn try_extend_into<C: ChunksMethods<Item = T>>(
        mut self,
        py: Python<'_>,
        list: &mut InnerList<C>,
    ) -> PyResult<()> {
        self.chunk_lens(list)
            .map(|len| {
                self.0
                    .by_ref()
                    .take(len)
                    .try_collect_with_size_hint_capacity::<PyResult<C::IntoChunk>>()
            })
            .try_for_each(|chunk| {
                list.push_chunk(py, chunk?);
                Ok(())
            })
    }
}

/// Where the items of a sorted batch go in a list: one location per item, in the same order.\
/// Computed without modifying the list, then applied from the last sublist to the first,
/// so splitting a sublist never shifts the ones left to fill.
pub(super) struct Batch<C: ChunksMethods> {
    items: SortedVec<C::Item>,
    idxs: LocIdxs,
}
impl<C: ChunksMethods> Batch<C> {
    pub fn new(items: SortedVec<C::Item>) -> Self {
        let idxs = Vec::with_capacity(items.len()).into();
        Self { items, idxs }
    }
    pub fn extend_into(mut self, py: Python<'_>, list: &mut InnerList<C>) -> PyResult<()> {
        let capacity = self.items.len();
        let mut chunk = self
            .items
            .into_iter()
            .map(|item| {
                self.idxs.locate(list, item.key())?;
                PyResult::Ok(item)
            })
            .try_collect_with_capacity::<PyResult<C::IntoChunk>>(capacity)?;
        list.len += self.idxs.len();
        self.idxs
            .chunk_by(|a, b| a.pos == b.pos)
            .rev()
            .for_each(|run| {
                list.chunks.drain_from(&mut chunk, run);
                // Once a split clears the index, the remaining runs leave it to be rebuilt on demand.
                if list.lengths.is_built() {
                    list.lengths.add_at(run[0].pos, run.len());
                }
                let mut last = run[0].pos;
                while list.chunks.keys()[last].len() > list.load.max_size() {
                    list.expand(py, last);
                    last += 1;
                }
                list.maxes[last] = list.chunks.keys()[last].last().clone_ref(py);
            });
        Ok(())
    }
}
#[derive(new)]
pub(super) struct Stream<C: ChunksMethods> {
    items: SortedVec<C::Item>,
    #[new(default)]
    curr: Loc,
}
impl<C: ChunksMethods> Stream<C> {
    /// Inserts the items into a non-empty list one by one, like `append`,
    /// but each item is only compared from the location of the previous one, and its key isn't computed again.
    pub fn extend_into(mut self, py: Python<'_>, list: &mut InnerList<C>) -> PyResult<()> {
        self.items.into_iter().try_for_each(|item| {
            let loc = list.next_sorted(self.curr, item.key())?;
            if loc.idx == list.chunks.keys()[loc.pos].len() {
                list.maxes[loc.pos] = item.key().clone().unbind();
            }
            list.insert_in_chunk(py, loc, item);
            list.len += 1;
            self.curr = loc;
            Ok(())
        })
    }
}
#[derive(From, Deref)]
struct LocIdxs(Vec<Loc>);
impl LocIdxs {
    /// Pushes the location of `key`, the next item of the batch.
    fn locate(
        &mut self,
        list: &InnerList<impl ChunksMethods>,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let loc = self
            .0
            .last()
            .copied()
            .unwrap_or_default()
            .pipe(|prev| list.next_sorted(prev, key))?;
        self.0.push(loc);
        Ok(())
    }
}
