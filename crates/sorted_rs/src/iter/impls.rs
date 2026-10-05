use crate::{
    Bounds, Loc,
    bounds::AtomicLoc,
    iter::{core::IterInner, traits::SortedNext},
};
use pyo3::{ffi, prelude::*};
use std::ptr;
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
        let (pos, idx) = self.1.load();
        let values = &unsafe { self.0.deref() }.values;
        values
            .get(pos)
            .and_then(|v| v.get(idx).map(|obj| (pos, idx, obj)))
            .or_else(|| {
                values
                    .get(pos + 1)
                    .and_then(|v| v.first())
                    .map(|obj| (pos + 1, 0, obj))
            })
            .map_or_else(ptr::null_mut, |(p, i, obj)| {
                let ptr = obj.as_ptr();
                unsafe { ffi::Py_INCREF(ptr) };
                self.1.store(p, i + 1);
                ptr
            })
    }
}

unsafe impl SortedNext for IterRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let (pos, idx) = self.1.load();
        let values = &unsafe { self.0.deref() }.values;
        idx.checked_sub(1)
            .map_or_else(
                || {
                    pos.checked_sub(1).and_then(|p| {
                        values
                            .get(p)
                            .and_then(|v| v.len().checked_sub(1))
                            .map(|i| (p, i))
                    })
                },
                |i| values.get(pos).filter(|v| i < v.len()).map(|_| (pos, i)),
            )
            .map_or_else(ptr::null_mut, |(p, i)| {
                self.1.store(p, i);
                unsafe { values.get_unchecked(p).pipe(|v| inc_ref_get(v, i)) }
            })
    }
}
unsafe impl SortedNext for IterBounded {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
        let (pos, idx) = self.1.load();
        let max = self.2;

        (pos < max.pos || (pos == max.pos && idx < max.idx))
            .then(|| data.values.get(pos))
            .flatten()
            .filter(|v| idx < v.len())
            .map_or_else(ptr::null_mut, |v| {
                if idx + 1 >= v.len() && pos < max.pos {
                    self.1.store(pos + 1, 0);
                } else {
                    self.1.store(pos, idx + 1);
                }
                unsafe { inc_ref_get(v, idx) }
            })
    }
}

unsafe impl SortedNext for IterBoundedRev {
    #[inline(always)]
    unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.0.deref() };
        let min = self.1;
        let (pos, idx) = self.2.load();

        (pos > min.pos || (pos == min.pos && idx > min.idx))
            .then(|| data.values.get(pos))
            .flatten()
            .filter(|v| idx <= v.len())
            .map_or_else(ptr::null_mut, |_| {
                let (pos, idx) = if idx == 0 {
                    let p = pos - 1;
                    (p, unsafe { data.values.get_unchecked(p) }.len() - 1)
                } else {
                    (pos, idx - 1)
                };
                if pos < min.pos || (pos == min.pos && idx < min.idx) {
                    ptr::null_mut()
                } else {
                    self.2.store(pos, idx);
                    unsafe { data.values.get_unchecked(pos).pipe(|v| inc_ref_get(v, idx)) }
                }
            })
    }
}
#[inline]
unsafe fn inc_ref_get(v: &[Py<PyAny>], idx: usize) -> *mut ffi::PyObject {
    let ptr = unsafe { v.get_unchecked(idx) }.as_ptr();
    unsafe { ffi::Py_INCREF(ptr) };
    ptr
}
