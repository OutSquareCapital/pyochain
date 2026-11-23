use crate::{
    bisect::bisect,
    bounds::Loc,
    sort::{SortKey, sorted::SortedVec},
    traits::ListsMethods,
    types::VecPy,
};

use derive_more::{Constructor, Deref, From};
use pyo3::{basic::CompareOp, prelude::*};
use pyo3_ext::prelude::*;
use std::{ptr, vec::Drain};

pub trait Chunk<T>: FromIterator<T> {
    fn with_capacity(capacity: usize) -> Self;
    fn append(&mut self, item: T);
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
    fn with_capacity(capacity: usize) -> Self {
        Vec::with_capacity(capacity)
    }
    fn append(&mut self, item: Py<PyAny>) {
        self.push(item);
    }
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        list.maxes.push(self.last().unwrap().clone_ref(py));
        list.keys_mut().push(self);
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
impl Chunk<(Py<PyAny>, Py<PyAny>)> for ChunkKey {
    fn with_capacity(capacity: usize) -> Self {
        Self::new(Vec::with_capacity(capacity), Vec::with_capacity(capacity))
    }
    fn append(&mut self, (key, value): (Py<PyAny>, Py<PyAny>)) {
        self.keys.push(key);
        self.values.push(value);
    }
    fn push_into(self, py: Python<'_>, list: &mut impl ListsMethods) {
        self.keys.push_into(py, list);
        list.values.push(self.values);
    }
    fn drain_into(&mut self, list: &mut impl ListsMethods, idxs: &[Loc]) {
        self.keys.drain_into(list, idxs);
        let start = self.values.len() - idxs.len();
        list.values[idxs[0].pos].insert_many(idxs, self.values.drain(start..));
    }
}

trait VecExt<T> {
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
    fn insert_many(&mut self, locs: &[Loc], items: Drain<'_, T>);
}
impl<T> VecExt<T> for Vec<T> {
    fn insert_many(&mut self, idxs: &[Loc], items: Drain<'_, T>) {
        let old_len = self.len();
        // `Drain` is a std `ExactSizeIterator`: the unsafe code below can trust its length.
        let new_len = old_len + items.len();
        assert_eq!(idxs.len(), items.len(), "one location is expected per item");
        assert!(
            idxs.is_sorted_by_key(|loc| loc.idx)
                && idxs.last().is_none_or(|last| last.idx <= old_len),
            "indices must be sorted and at most the length"
        );
        self.reserve(items.len());
        let base = self.as_mut_ptr();
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
                ptr::copy(base.add(idx), base.add(dst_end + 1), count);
                ptr::write(base.add(dst_end), item);
            }
            src_end = idx;
        });
        // SAFETY: every slot of `[0, new_len)` is now initialized exactly once:
        // `[0, src_end)` never moved, and `dst_end == src_end` once all items are written.
        unsafe { self.set_len(new_len) };
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
        let mut chunk = L::Chunk::with_capacity(self.items.len());
        self.items.into_iter().try_for_each(|item| {
            self.idxs.locate(list, item.key())?;
            chunk.append(item);
            PyResult::Ok(())
        })?;
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
                while list.keys()[last].len() > list.load.higher_limit() {
                    list.expand(py, last);
                    last += 1;
                }
                list.maxes[last] = list.keys()[last].last().unwrap().clone_ref(py);
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
#[cfg(test)]
mod tests {
    use super::VecExt;
    use crate::bounds::Loc;

    fn locs(idxs: &[usize]) -> Vec<Loc> {
        idxs.iter().copied().map(Loc::with_idx).collect()
    }
    /// Reference implementation: plain `insert`s, from the last item to the first.
    fn naive(mut vec: Vec<String>, idxs: &[usize], items: Vec<String>) -> Vec<String> {
        idxs.iter()
            .zip(items)
            .rev()
            .for_each(|(&idx, item)| vec.insert(idx, item));
        vec
    }
    fn strings(prefix: &str, len: usize) -> Vec<String> {
        (0..len).map(|i| format!("{prefix}{i}")).collect()
    }

    #[test]
    fn matches_naive_inserts() {
        let cases: [(usize, &[usize]); 7] = [
            (0, &[]),
            (0, &[0, 0, 0]),
            (5, &[]),
            (5, &[0]),
            (5, &[5, 5]),
            (5, &[0, 2, 2, 4, 5]),
            (8, &[1, 1, 3, 3, 3, 7, 8, 8]),
        ];
        for (len, idxs) in cases {
            let mut items = strings("new", idxs.len());
            let expected = naive(strings("old", len), idxs, items.clone());
            let mut vec = strings("old", len);
            vec.insert_many(&locs(idxs), items.drain(..));
            assert_eq!(vec, expected, "len={len} idxs={idxs:?}");
            assert!(items.is_empty(), "the source buffer must be left empty");
        }
    }
    #[test]
    #[should_panic(expected = "indices must be sorted")]
    fn rejects_unsorted_indices() {
        strings("old", 3).insert_many(&locs(&[2, 1]), strings("new", 2).drain(..));
    }
    #[test]
    #[should_panic(expected = "indices must be sorted")]
    fn rejects_out_of_bounds_index() {
        strings("old", 3).insert_many(&locs(&[4]), strings("new", 1).drain(..));
    }
}
