use crate::{
    Bounds, Loc,
    bounds::AtomicLoc,
    iter::{core::IterInner, traits::SortedNext},
};
use pyo3::ffi;
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

#[derive(Debug)]
pub struct Iter(IterInner, AtomicLoc);

#[derive(Debug)]
pub struct IterRev(IterInner, AtomicLoc);

#[derive(Debug)]
pub struct IterBounded(IterInner, AtomicLoc, u64);

#[derive(Debug)]
pub struct IterBoundedRev(IterInner, u64, AtomicLoc);

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
        let pos = unsafe { inner.deref() }
            .values
            .len()
            .pipe(Loc::with_pos)
            .into();
        Self(inner, pos)
    }
}

impl IterBounded {
    #[inline]
    pub fn new(owner: impl Into<IterInner>, bounds: Bounds) -> Self {
        Self(owner.into(), bounds.min.into(), bounds.max.into())
    }
}

impl IterBoundedRev {
    #[inline]
    pub fn new(owner: impl Into<IterInner>, bounds: Bounds) -> Self {
        Self(owner.into(), bounds.min.into(), bounds.max.into())
    }
}

unsafe impl SortedNext for Iter {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next_fwd::<false>(&self.1, 0) }
    }
}

unsafe impl SortedNext for IterRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next_rev(&self.1, 0) }
    }
}
unsafe impl SortedNext for IterBounded {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next_fwd::<true>(&self.1, self.2) }
    }
}

unsafe impl SortedNext for IterBoundedRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next_rev(&self.2, self.1) }
    }
}
