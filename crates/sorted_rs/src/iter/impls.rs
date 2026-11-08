use crate::{
    Bounds, ListGetter,
    bounds::Loc,
    iter::{
        core::InnerRef,
        cursor::Cursor,
        traits::{Item, SortedNext},
    },
};
use derive_more::{Constructor, Deref};
use pyo3::{ffi, prelude::*};

use tap::prelude::*;
type IntoBounded<'a, 'py, T> = (&'a Bound<'py, T>, Bounds);
#[derive(Debug, Constructor)]
pub struct Iter<const FWD: bool> {
    pyref: InnerRef,
    cursor: Cursor,
}

#[derive(Debug, Deref, Constructor)]
pub struct IterBounded<const FWD: bool> {
    #[deref]
    inner: Iter<FWD>,
    limit: u64,
}

impl<T: ListGetter> From<&Bound<'_, T>> for Iter<true> {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        Self::new(InnerRef::new(owner), Cursor::default())
    }
}

impl<T: ListGetter> From<&Bound<'_, T>> for Iter<false> {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        let pyref = InnerRef::new(owner);
        let cursor = pyref.values.len().pipe(Loc::with_pos).into();
        Self::new(pyref, cursor)
    }
}

impl<T: ListGetter> From<IntoBounded<'_, '_, T>> for IterBounded<true> {
    fn from((owner, bounds): IntoBounded<'_, '_, T>) -> Self {
        let inner = Iter::new(InnerRef::new(owner), bounds.min.into());
        Self::new(inner, bounds.max.into())
    }
}

impl<T: ListGetter> From<IntoBounded<'_, '_, T>> for IterBounded<false> {
    #[inline]
    fn from((owner, bounds): IntoBounded<'_, '_, T>) -> Self {
        let inner = Iter::new(InnerRef::new(owner), bounds.max.into());
        Self::new(inner, bounds.min.into())
    }
}
impl<const FWD: bool> Iter<FWD> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.pyref.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pyref.is_empty()
    }
    #[inline(always)]
    fn inner_next(&self) -> Option<(u64, &Py<PyAny>)> {
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

unsafe impl<const FWD: bool> SortedNext for Iter<FWD> {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next().map(|(at, item)| self.yield_ptr(item, at))
    }
}
unsafe impl<const FWD: bool> SortedNext for IterBounded<FWD> {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next()
            .filter(|&(at, _)| {
                at == self.limit || (FWD && at < self.limit) || (!FWD && at > self.limit)
            })
            .map(|(at, item)| self.inner.yield_ptr(item, at))
    }
}
