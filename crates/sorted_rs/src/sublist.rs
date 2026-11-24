use std::vec::Drain;

use derive_more::{Deref, IntoIterator};
use pyo3::prelude::*;

use crate::{bounds::Loc, types::VecPy};

/// A sublist of a sorted list. Never empty.\
/// Only gives access to its items as a slice, so its length can only change through the methods below,
/// which never leave it empty.
#[derive(Debug, Deref, IntoIterator)]
#[deref(forward)]
#[into_iterator(ref)]
pub struct SubList(VecPy);
impl From<VecPy> for SubList {
    fn from(vec: VecPy) -> Self {
        debug_assert!(!vec.is_empty());
        Self(vec)
    }
}
impl SubList {
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
