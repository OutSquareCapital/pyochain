use std::ptr;

use crate::{
    ListGetter,
    iter::{Iter, cursor::CursorMethods},
    sublist::{Chunks, ChunksMethods},
};
use derive_more::Constructor;
use pyo3::prelude::*;
/// Wrapper type to hold pointers to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug, Constructor)]
pub(super) struct InnerRef {
    /// Pointer to the values of the sorted collection.
    values: *const Chunks,
    /// Pointer to the length of the sorted collection.
    len: *const usize,
    /// Keeps the collection alive so `values` and `len` stay valid.
    owner: Py<PyAny>,
}
impl InnerRef {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self::new(self.values, self.len, self.owner.clone_ref(py))
    }
    pub fn from_owner<T: ListGetter>(value: &Bound<'_, T>, pyref: &T) -> Self {
        let owner = value.clone().into_any().unbind();
        let guard = pyref.read();
        Self::new(
            ptr::from_ref(guard.chunks.values()),
            ptr::from_ref(&guard.len),
            owner,
        )
    }
    #[inline(always)]
    pub fn values(&self) -> &Chunks {
        unsafe { &*self.values }
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        unsafe { *self.len }
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_iterator<const FWD: bool, T: CursorMethods>(self, cursor: T) -> Iter<FWD, T> {
        Iter::new(self, cursor)
    }
}
unsafe impl Send for InnerRef {}
unsafe impl Sync for InnerRef {}
