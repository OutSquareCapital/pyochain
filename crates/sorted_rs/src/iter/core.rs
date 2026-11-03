use std::{ops::Deref, ptr};

use crate::{ListGetter, inner::InnerData, iter::cursor::Cursor};
use derive_more::{Constructor, Deref};
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
impl<T: ListGetter> From<&Bound<'_, T>> for InnerRef {
    fn from(value: &Bound<'_, T>) -> Self {
        value
            .get()
            .read()
            .deref()
            .deref()
            .pipe(ptr::from_ref::<InnerData>)
            .pipe(|data| Self::new(data, value.clone().into_any().unbind()))
    }
}
#[derive(Debug, Constructor)]
pub struct InnerIter {
    pyref: InnerRef,
    cursor: Cursor,
}
#[derive(Debug, Constructor, Deref)]
pub struct InnerIterBounded {
    #[deref]
    inner: InnerIter,
    limit: u64,
}
impl InnerIter {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.pyref.len
    }
    #[inline(always)]
    pub(super) unsafe fn next<const FWD: bool>(&self) -> *mut ffi::PyObject {
        self.inner_next::<FWD>()
            .map_or_else(ptr::null_mut, |(at, item)| self.yield_ptr(item, at))
    }
    #[inline(always)]
    fn inner_next<const FWD: bool>(&self) -> Option<(u64, &Py<PyAny>)> {
        let values = &self.pyref.values;
        let (at, pos, idx) = self.cursor.load();
        let current = values.get(pos);
        if FWD {
            current
                .and_then(|v| v.get(idx))
                .map(|obj| (at + 1, obj))
                .or_else(|| {
                    let p = pos + 1;
                    values
                        .get(p)?
                        .first()
                        .map(|obj| (Cursor::new_pos(p) + 1, obj))
                })
        } else {
            current
                .and_then(|v| v.get(idx.checked_sub(1)?))
                .map(|obj| (at - 1, obj))
                .or_else(|| {
                    let p = pos.checked_sub(1)?;
                    let v = values.get(p)?;
                    v.last()
                        .map(|obj| (Cursor::new_pos(p) | (v.len() - 1) as u64, obj))
                })
        }
    }
    fn yield_ptr(&self, item: &Py<PyAny>, at: u64) -> *mut ffi::PyObject {
        self.cursor.store(at);
        let ptr = item.as_ptr();
        unsafe { ffi::Py_INCREF(ptr) };
        ptr
    }
}
impl InnerIterBounded {
    #[inline(always)]
    pub(super) unsafe fn next<const FWD: bool>(&self) -> *mut ffi::PyObject {
        self.inner_next::<FWD>()
            .filter(|&(at, _)| {
                at == self.limit || (FWD && at < self.limit) || (!FWD && at > self.limit)
            })
            .map_or_else(ptr::null_mut, |(at, item)| self.inner.yield_ptr(item, at))
    }
}
