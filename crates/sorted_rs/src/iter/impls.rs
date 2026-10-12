use crate::{
    Bounds, Loc,
    iter::{
        core::{InnerIter, InnerRef},
        cursor::Cursor,
        traits::SortedNext,
    },
};
use derive_more::{Deref, From};

use pyo3::ffi;
use tap::prelude::*;

pub enum IterKind {
    Fwd,
    Bwd,
    Bounded(Bounds, bool),
}

#[derive(Debug, From, Deref)]
pub struct Iter(InnerIter);

#[derive(Debug, From, Deref)]
pub struct IterRev(InnerIter);

#[derive(Debug, Deref)]
pub struct IterBounded(#[deref] InnerIter, u64);

#[derive(Debug, Deref)]
pub struct IterBoundedRev(#[deref] InnerIter, u64);

impl Iter {
    #[inline]
    pub fn new(owner: impl Into<InnerRef>) -> Self {
        InnerIter::new(owner.into(), Cursor::default()).into()
    }
}

impl IterRev {
    #[inline]
    pub fn new(owner: impl Into<InnerRef>) -> Self {
        let inner = owner.into();
        let pos = inner.values.len().pipe(Loc::with_pos).into();
        InnerIter::new(inner, pos).into()
    }
}

impl IterBounded {
    #[inline]
    pub fn new(owner: impl Into<InnerRef>, bounds: Bounds) -> Self {
        let inner = InnerIter::new(owner.into(), bounds.min.into());
        Self(inner, bounds.max.into())
    }
}

impl IterBoundedRev {
    #[inline]
    pub fn new(owner: impl Into<InnerRef>, bounds: Bounds) -> Self {
        let inner = InnerIter::new(owner.into(), bounds.max.into());
        Self(inner, bounds.min.into())
    }
}

unsafe impl SortedNext for Iter {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next::<false, true>(0) }
    }
}

unsafe impl SortedNext for IterRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next::<false, false>(0) }
    }
}
unsafe impl SortedNext for IterBounded {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next::<true, true>(self.1) }
    }
}

unsafe impl SortedNext for IterBoundedRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        unsafe { self.0.next::<true, false>(self.1) }
    }
}
