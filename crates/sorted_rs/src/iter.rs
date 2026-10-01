use crate::{Bounds, Loc, bounds::AtomicLoc, inner::InnerData};
use derive_more::From;
use parking_lot::{RawRwLock, lock_api::RwLockReadGuard};
use pyo3::{ffi, prelude::*};
use pyo3_ext::prelude::*;
use std::{ops::Deref, ptr};
use tap::prelude::*;
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
/// Common iterator interface for all sorted iterators.
/// # Safety
/// All implementers must ensure that all pointers are correctly managed,\
/// and that all call sites are restricted to contexts with a valid Python interpreter.
pub unsafe trait SortedNext {
    /// Concrete logic of the python `Iterator`.
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    unsafe fn next(&self) -> *mut ffi::PyObject;
}
/// Trait for the Pyclasses that effectively wrap implementors of `SortedNext`.\
/// You can consider `SortedNext` as the "logic provider" from rust and `PySortedIter` as the "interface provider" for Python.
pub trait PySortedIter<T>: PyFrozenClass + Deref<Target = T>
where
    T: SortedNext,
{
    /// The method that is effectively called by Python as the `tp_iternext` body.
    /// # Safety
    /// The caller must ensure that:
    /// - the Iterator is wrapped in a pyclass
    /// - the function is called strictly from a context with a valid Python interpreter
    #[inline(always)]
    unsafe extern "C" fn tp_iternext(obj: *mut ffi::PyObject) -> *mut ffi::PyObject {
        unsafe {
            let py = Python::assume_attached();
            Borrowed::from_ptr(py, obj)
                .cast_unchecked::<Self>()
                .get()
                .next()
        }
    }
    fn install(py: Python<'_>) {
        unsafe {
            let ty = Self::type_object_raw(py);
            (*ty).tp_iternext = Some(Self::tp_iternext);
            ffi::PyType_Modified(ty);
        }
    }
}

/// Wrapper type to hold a pointer to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug, From)]
pub struct IterInner(*const InnerData);

unsafe impl Send for IterInner {}
unsafe impl Sync for IterInner {}
impl<T> From<RwLockReadGuard<'_, RawRwLock, T>> for IterInner
where
    T: Deref<Target = InnerData> + Send + Sync + 'static,
{
    fn from(guard: RwLockReadGuard<'_, RawRwLock, T>) -> Self {
        guard
            .deref()
            .deref()
            .pipe(ptr::from_ref::<T::Target>)
            .into()
    }
}
impl IterInner {
    #[inline(always)]
    unsafe fn deref(&self) -> &InnerData {
        unsafe { &*self.0 }
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

impl<T> From<T> for Iter
where
    T: Into<IterInner>,
{
    fn from(owner: T) -> Self {
        Self(owner.into(), AtomicLoc::default())
    }
}

impl<T> From<T> for IterRev
where
    T: Into<IterInner>,
{
    fn from(owner: T) -> Self {
        let inner = owner.into();
        let data = unsafe { inner.deref() };
        let pos = data.values.len().saturating_sub(1);
        let idx = data.values.last().map_or(0, Vec::len);
        let loc = Loc::new(pos, idx).into();
        Self(inner, loc)
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

unsafe impl SortedNext for Iter {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
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
            ptr::null_mut()
        }
    }
}

unsafe impl SortedNext for IterRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
        let (pos, idx) = self.1.load();

        if pos == 0 && idx == 0 {
            ptr::null_mut()
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

unsafe impl SortedNext for IterBounded {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
        let (pos, idx) = self.1.load();
        let max = self.2;

        if pos == max.pos && idx == max.idx {
            ptr::null_mut()
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

unsafe impl SortedNext for IterBoundedRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
        let min = self.1;
        let (pos, idx) = self.2.load();

        if pos == min.pos && idx == min.idx {
            ptr::null_mut()
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
