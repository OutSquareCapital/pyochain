use crate::{
    Bounds, ListGetter,
    bounds::Loc,
    iter::{
        core::{InnerIter, InnerIterBounded, InnerRef},
        cursor::Cursor,
        traits::{Item, SortedNext},
    },
};
use derive_more::Deref;
use pyo3::prelude::*;

use tap::prelude::*;
type IntoBounded<'a, 'py, T> = (&'a Bound<'py, T>, Bounds);
#[derive(Debug, Deref)]
pub struct Iter(InnerIter);

#[derive(Debug, Deref)]
pub struct IterRev(InnerIter);

#[derive(Debug, Deref)]
pub struct IterBounded(InnerIterBounded);

#[derive(Debug, Deref)]
pub struct IterBoundedRev(InnerIterBounded);

impl<T: ListGetter> From<&Bound<'_, T>> for Iter {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        Self(InnerIter::new(owner.into(), Cursor::default()))
    }
}

impl<T: ListGetter> From<&Bound<'_, T>> for IterRev {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        let pyref = owner.conv::<InnerRef>();
        let cursor = pyref.values.len().pipe(Loc::with_pos).into();
        Self(InnerIter::new(pyref, cursor))
    }
}

impl<T: ListGetter> From<IntoBounded<'_, '_, T>> for IterBounded {
    fn from((owner, bounds): IntoBounded<'_, '_, T>) -> Self {
        let inner = InnerIter::new(owner.into(), bounds.min.into());
        Self(InnerIterBounded::new(inner, bounds.max.into()))
    }
}

impl<T: ListGetter> From<IntoBounded<'_, '_, T>> for IterBoundedRev {
    #[inline]
    fn from((owner, bounds): IntoBounded<'_, '_, T>) -> Self {
        let inner = InnerIter::new(owner.into(), bounds.max.into());
        Self(InnerIterBounded::new(inner, bounds.min.into()))
    }
}
unsafe impl SortedNext for Iter {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        unsafe { self.0.next::<true>() }
    }
}
unsafe impl SortedNext for IterRev {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        unsafe { self.0.next::<false>() }
    }
}
unsafe impl SortedNext for IterBounded {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        unsafe { self.0.next::<true>() }
    }
}
unsafe impl SortedNext for IterBoundedRev {
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        unsafe { self.0.next::<false>() }
    }
}
