use std::{ops::Deref, ptr};

use crate::{bounds::AtomicLoc, inner::InnerData};
use derive_more::Constructor;
use parking_lot::{RawRwLock, lock_api::RwLockReadGuard};
use pyo3::{ffi, prelude::*};
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
    #[inline(always)]
    pub(super) unsafe fn next_fwd<const BOUNDED: bool>(
        &self,
        cur: &AtomicLoc,
        end: u64,
    ) -> *mut ffi::PyObject {
        let values = &unsafe { self.deref() }.values;
        let (at, pos, idx) = cur.load();
        values
            .get(pos)
            .and_then(|v| v.get(idx))
            .map(|obj| (at, obj))
            .or_else(|| {
                values
                    .get(pos + 1)?
                    .first()
                    .map(|obj| ((pos as u64 + 1) << u32::BITS, obj))
            })
            .filter(|&(at, _)| !BOUNDED || at < end)
            .map_or_else(ptr::null_mut, |(at, item)| {
                cur.store(at + 1);
                unsafe { yield_item(item) }
            })
    }

    /// Element before `cur`, or the last one of the previous sublist, if located at or after `first`.
    #[inline(always)]
    pub(super) unsafe fn next_rev(&self, cur: &AtomicLoc, first: u64) -> *mut ffi::PyObject {
        let values = &unsafe { self.deref() }.values;
        let (at, pos, idx) = cur.load();
        idx.checked_sub(1)
            .map_or_else(
                || {
                    let p = pos.checked_sub(1)?;
                    let v = values.get(p)?;
                    v.last()
                        .map(|obj| (((p as u64) << u32::BITS) | (v.len() - 1) as u64, obj))
                },
                |i| values.get(pos)?.get(i).map(|obj| (at - 1, obj)),
            )
            .filter(|&(at, _)| at >= first)
            .map_or_else(ptr::null_mut, |(at, item)| {
                cur.store(at);
                unsafe { yield_item(item) }
            })
    }
}
#[inline]
unsafe fn yield_item(item: &Py<PyAny>) -> *mut ffi::PyObject {
    let ptr = item.as_ptr();
    unsafe { ffi::Py_INCREF(ptr) };
    ptr
}
