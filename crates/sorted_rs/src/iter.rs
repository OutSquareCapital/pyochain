use crate::{Bounds, Loc, bounds::AtomicLoc, inner::InnerData};
use derive_more::Constructor;
use parking_lot::RwLock;
use pyo3::ffi;
use std::{any::Any, ops::Deref, sync::Arc};

pub enum IterKind {
    Fwd,
    Rev,
    Bounded(Bounds),
    BoundedRev(Bounds),
}

impl IterKind {
    #[inline]
    #[must_use]
    pub fn from_bounds(bounds: Option<Bounds>, rev: bool) -> Self {
        let bounds = bounds.unwrap_or_default();
        if rev {
            Self::BoundedRev(bounds)
        } else {
            Self::Bounded(bounds)
        }
    }
}

#[derive(Constructor, Debug)]
pub struct IterInner {
    _owner: Arc<dyn Any + Send + Sync>,
    data: *const InnerData,
}

unsafe impl Send for IterInner {}
unsafe impl Sync for IterInner {}

impl<T> From<Arc<RwLock<T>>> for IterInner
where
    T: Deref<Target = InnerData> + Send + Sync + 'static,
{
    fn from(owner: Arc<RwLock<T>>) -> Self {
        let guard = owner.read();
        let data = std::ptr::from_ref::<InnerData>(&**guard);
        drop(guard);
        Self::new(owner, data)
    }
}

impl IterInner {
    #[inline(always)]
    unsafe fn get(&self) -> &InnerData {
        unsafe { &*self.data }
    }
}

#[derive(Debug)]
pub struct Iter(IterInner, AtomicLoc);

#[derive(Debug)]
pub struct IterRev(IterInner, AtomicLoc);

#[derive(Debug)]
pub struct IterBounded(IterInner, AtomicLoc, Loc);

#[derive(Debug)]
pub struct IterBoundedRev(IterInner, Loc, AtomicLoc);

impl<T> From<Arc<RwLock<T>>> for Iter
where
    T: Deref<Target = InnerData> + Send + Sync + 'static,
{
    fn from(owner: Arc<RwLock<T>>) -> Self {
        Self(owner.into(), AtomicLoc::default())
    }
}

impl<T> From<Arc<RwLock<T>>> for IterRev
where
    T: Deref<Target = InnerData> + Send + Sync + 'static,
{
    fn from(owner: Arc<RwLock<T>>) -> Self {
        let (data, pos, idx) = {
            let guard = owner.read();
            let pos = guard.values.len().saturating_sub(1);
            let idx = guard.values.last().map_or(0, Vec::len);
            (std::ptr::from_ref::<InnerData>(&**guard), pos, idx)
        };
        Self(
            IterInner::new(owner, data),
            AtomicLoc::new(pos.into(), idx.into()),
        )
    }
}

impl IterBounded {
    #[inline]
    pub fn new(owner: impl Into<IterInner>, bounds: Bounds) -> Self {
        Self(owner.into(), bounds.min.into(), bounds.max)
    }
}

impl IterBoundedRev {
    #[inline]
    pub fn new(owner: impl Into<IterInner>, bounds: Bounds) -> Self {
        Self(owner.into(), bounds.min, bounds.max.into())
    }
}

impl Iter {
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    #[inline(always)]
    pub unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.get() };
        let (pos, idx) = self.1.load();

        if let Some(v) = data.values.get(pos) {
            let item = unsafe { v.get_unchecked(idx) };
            let ptr = item.as_ptr();
            unsafe { ffi::Py_INCREF(ptr) };

            if idx + 1 < v.len() {
                self.1.store(pos, idx + 1);
            } else {
                self.1.store(pos + 1, 0);
            }
            ptr
        } else {
            std::ptr::null_mut()
        }
    }
}

impl IterRev {
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    #[inline(always)]
    pub unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.get() };
        let (pos, idx) = self.1.load();

        if pos == 0 && idx == 0 {
            std::ptr::null_mut()
        } else {
            let (pos, idx) = if idx == 0 {
                let p = pos - 1;
                (p, unsafe { data.values.get_unchecked(p) }.len() - 1)
            } else {
                (pos, idx - 1)
            };

            let item = unsafe { data.values.get_unchecked(pos).get_unchecked(idx) };
            let ptr = item.as_ptr();
            unsafe { ffi::Py_INCREF(ptr) };

            self.1.store(pos, idx);
            ptr
        }
    }
}

impl IterBounded {
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    #[inline(always)]
    pub unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.get() };
        let (pos, idx) = self.1.load();
        let max = self.2;

        if pos == max.pos && idx == max.idx {
            std::ptr::null_mut()
        } else {
            let v = unsafe { data.values.get_unchecked(pos) };
            let item = unsafe { v.get_unchecked(idx) };
            let ptr = item.as_ptr();
            unsafe { ffi::Py_INCREF(ptr) };

            if idx + 1 >= v.len() && pos < max.pos {
                self.1.store(pos + 1, 0);
            } else {
                self.1.store(pos, idx + 1);
            }
            ptr
        }
    }
}

impl IterBoundedRev {
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    #[inline(always)]
    pub unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.get() };
        let min = self.1;
        let (pos, idx) = self.2.load();

        if pos == min.pos && idx == min.idx {
            std::ptr::null_mut()
        } else {
            let (pos, idx) = if idx == 0 {
                let p = pos - 1;
                (p, unsafe { data.values.get_unchecked(p) }.len() - 1)
            } else {
                (pos, idx - 1)
            };

            let item = unsafe { data.values.get_unchecked(pos).get_unchecked(idx) };
            let ptr = item.as_ptr();
            unsafe { ffi::Py_INCREF(ptr) };

            self.2.store(pos, idx);
            ptr
        }
    }
}
