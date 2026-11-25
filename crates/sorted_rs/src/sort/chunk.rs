use crate::{
    bisect::bisect, bounds::Loc, item::Item, sort::sorted::SortedVec, sublist::SubList,
    traits::ListsMethods, types::VecPy,
};

use derive_more::{Constructor, Deref, From};
use pyo3::{basic::CompareOp, prelude::*};
use pyo3_ext::prelude::*;
use std::iter::RepeatN;
use std_tools::{collections::WithCapacity, prelude::TryIterator};

pub trait Chunk<T>: WithCapacity<T> {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods);
    /// Moves the last `run.len()` items of the chunk into the sublist of the run,
    /// each item at the matching location of `run`.
    fn drain_into(&mut self, list: &mut impl ListsMethods, idxs: &[Loc]);
}
#[derive(Constructor, From)]
pub struct ChunkKey {
    keys: VecPy,
    values: VecPy,
}
impl Chunk<Py<PyAny>> for VecPy {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        let sublist = SubList::from(self);
        list.maxes.push(sublist.last().clone_ref(py));
        list.keys_mut().push(sublist);
    }
    fn drain_into(&mut self, list: &mut impl ListsMethods, idxs: &[Loc]) {
        let start = self.len() - idxs.len();
        list.keys_mut()[idxs[0].pos].insert_many(idxs, self.drain(start..));
    }
}
impl FromIterator<(Py<PyAny>, Py<PyAny>)> for ChunkKey {
    fn from_iter<I: IntoIterator<Item = (Py<PyAny>, Py<PyAny>)>>(iter: I) -> Self {
        <(Vec<_>, Vec<_>)>::from_iter(iter).into()
    }
}
impl WithCapacity<(Py<PyAny>, Py<PyAny>)> for ChunkKey {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }
    fn push(&mut self, (key, value): (Py<PyAny>, Py<PyAny>)) {
        self.keys.push(key);
        self.values.push(value);
    }
}
impl Chunk<(Py<PyAny>, Py<PyAny>)> for ChunkKey {
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        self.keys.push_into(py, list);
        list.values.push(self.values.into());
    }
    fn drain_into(&mut self, list: &mut impl ListsMethods, idxs: &[Loc]) {
        self.keys.drain_into(list, idxs);
        let start = self.values.len() - idxs.len();
        list.values[idxs[0].pos].insert_many(idxs, self.values.drain(start..));
    }
}
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
    pub fn extend_into<L: ListsMethods<Item = I::Item>>(mut self, py: Python<'_>, list: &mut L) {
        self.chunk_lens(list)
            .map(|len| self.0.by_ref().take(len).collect::<L::Chunk>())
            .for_each(|chunk| chunk.push_into(py, list));
    }
    fn chunk_lens(&self, list: &mut impl ListsMethods) -> RepeatN<usize> {
        debug_assert!(list.is_empty());
        let length = self.0.len();
        let load = *list.load;
        list.len = length;
        std::iter::repeat_n(load, length.div_ceil(load))
    }
}
impl<T, I: ExactSizeIterator<Item = PyResult<T>>> Chunked<I> {
    pub fn try_extend_into<L: ListsMethods<Item = T>>(
        mut self,
        py: Python<'_>,
        list: &mut L,
    ) -> PyResult<()> {
        self.chunk_lens(list)
            .map(|len| {
                self.0
                    .by_ref()
                    .take(len)
                    .try_collect_with_size_hint_capacity::<PyResult<L::Chunk>>()
            })
            .try_for_each(|chunk| {
                chunk?.push_into(py, list);
                Ok(())
            })
    }
}

/// Where the items of a sorted batch go in a list: one location per item, in the same order.\
/// Computed without modifying the list, then applied from the last sublist to the first,
/// so splitting a sublist never shifts the ones left to fill.
pub(super) struct Batch<L: ListsMethods> {
    items: SortedVec<L::Item>,
    idxs: LocIdxs,
}
impl<L: ListsMethods> Batch<L> {
    pub fn new(items: SortedVec<L::Item>) -> Self {
        let idxs = Vec::with_capacity(items.len()).into();
        Self { items, idxs }
    }
    pub fn extend_into(mut self, py: Python<'_>, list: &mut L) -> PyResult<()> {
        let capacity = self.items.len();
        let mut chunk = self
            .items
            .into_iter()
            .map(|item| {
                self.idxs.locate(list, item.key())?;
                PyResult::Ok(item)
            })
            .try_collect_with_capacity::<PyResult<L::Chunk>>(capacity)?;
        list.len += self.idxs.len();
        self.idxs
            .chunk_by(|a, b| a.pos == b.pos)
            .rev()
            .for_each(|run| {
                chunk.drain_into(list, run);
                // Once a split clears the index, the remaining runs leave it to be rebuilt on demand.
                if list.lengths.is_built() {
                    list.lengths.add_at(run[0].pos, run.len());
                }
                let mut last = run[0].pos;
                while list.keys()[last].len() > list.load.max_size() {
                    list.expand(py, last);
                    last += 1;
                }
                list.maxes[last] = list.keys()[last].last().clone_ref(py);
            });
        Ok(())
    }
}
#[derive(From, Deref)]
struct LocIdxs(Vec<Loc>);
impl LocIdxs {
    /// Pushes the location of `key`, the next item of the batch.
    fn locate<L: ListsMethods>(&mut self, list: &L, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let loc = self
            .0
            .last()
            .copied()
            .unwrap_or_default()
            .next_sorted(list, key)?;
        self.0.push(loc);
        Ok(())
    }
}
impl Loc {
    /// Location of `key`, the item following the one at `self` in a sorted batch.\
    /// Only compares from `self`, since `key` can't go before it.
    /// Stays valid if the previous item was already inserted at `self`, and its sublist split.\
    /// Uses `bisect_right` semantics like `append`: a key equal to a max goes into the next sublist,
    /// and a key greater than every max goes into the last one.
    pub(super) fn next_sorted(
        self,
        list: &impl ListsMethods,
        key: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let last = list.maxes.len() - 1;
        let pos = if self.pos == last
            || key.rich_compare_bool(list.maxes[self.pos].bind(key.py()), CompareOp::Lt)?
        {
            self.pos
        } else {
            bisect::<false>(&list.maxes[self.pos + 1..last], key)? + self.pos + 1
        };
        let low = if pos == self.pos { self.idx } else { 0 };
        let idx = bisect::<false>(&list.keys()[pos][low..], key)?;
        Ok(Self::new(pos, low + idx))
    }
}
