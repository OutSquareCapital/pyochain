use crate::inner::InnerData;
use derive_more::Constructor;
use parking_lot::{RawRwLock, lock_api::RwLockReadGuard};
use pyo3::prelude::*;
use std::{ops::Deref, ptr};
use tap::prelude::*;

/// Wrapper type to hold a pointer to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug, Constructor)]
pub struct IterInner {
    /// Pointer to the `InnerData` of the sorted collection.
    data: *const InnerData,
    /// Keeps the collection alive so `data` stays valid.
    _owner: Py<PyAny>,
}

unsafe impl Send for IterInner {}
unsafe impl Sync for IterInner {}
impl<T> From<(RwLockReadGuard<'_, RawRwLock, T>, Py<PyAny>)> for IterInner
where
    T: Deref<Target = InnerData> + Send + Sync + 'static,
{
    fn from((guard, owner): (RwLockReadGuard<'_, RawRwLock, T>, Py<PyAny>)) -> Self {
        guard
            .deref()
            .deref()
            .pipe(ptr::from_ref::<T::Target>)
            .pipe(|data| Self::new(data, owner))
    }
}
impl IterInner {
    #[inline(always)]
    pub(super) unsafe fn deref(&self) -> &InnerData {
        unsafe { &*self.data }
    }
}
