use std::{ops::Deref, ptr};

use crate::{inner::InnerData, iter::cursor::Cursor};
use derive_more::Constructor;
use parking_lot::{RawRwLock, lock_api::RwLockReadGuard};
use pyo3::{ffi, prelude::*};
use tap::prelude::*;
/// Wrapper type to hold a pointer to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug, Constructor)]
pub struct InnerRef {
    /// Pointer to the `InnerData` of the sorted collection.
    data: *const InnerData,
    /// Keeps the collection alive so `data` stays valid.
    _owner: Py<PyAny>,
}
impl Deref for InnerRef {
    type Target = InnerData;
    #[inline(always)]
    fn deref(&self) -> &InnerData {
        unsafe { &*self.data }
    }
}

unsafe impl Send for InnerRef {}
unsafe impl Sync for InnerRef {}
impl<T> From<(RwLockReadGuard<'_, RawRwLock, T>, Py<PyAny>)> for InnerRef
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
#[derive(Debug, Constructor)]
pub struct InnerIter {
    inner: InnerRef,
    cursor: Cursor,
}
impl InnerIter {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.inner.len
    }
    #[inline(always)]
    pub(super) unsafe fn next<const BOUNDED: bool, const FWD: bool>(
        &self,
        limit: u64,
    ) -> *mut ffi::PyObject {
        let values = &self.inner.values;
        let (at, pos, idx) = self.cursor.load();
        if FWD {
            values
                .get(pos)
                .and_then(|v| v.get(idx))
                .map(|obj| (at + 1, obj))
                .or_else(|| {
                    values
                        .get(pos + 1)?
                        .first()
                        .map(|obj| (((pos as u64 + 1) << u32::BITS) + 1, obj))
                })
        } else {
            idx.checked_sub(1).map_or_else(
                || {
                    let p = pos.checked_sub(1)?;
                    let v = values.get(p)?;
                    v.last()
                        .map(|obj| (((p as u64) << u32::BITS) | (v.len() - 1) as u64, obj))
                },
                |i| values.get(pos)?.get(i).map(|obj| (at - 1, obj)),
            )
        }
        .filter(|&(at, _)| !BOUNDED || at == limit || (FWD && at < limit) || (!FWD && at > limit))
        .map_or_else(ptr::null_mut, |(at, item)| {
            self.cursor.store(at);
            let ptr = item.as_ptr();
            unsafe { ffi::Py_INCREF(ptr) };
            ptr
        })
    }
}
