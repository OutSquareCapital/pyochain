use crate::{
    ListGetter,
    bounds::Loc,
    iter::{
        core::InnerRef,
        cursor::{BoundedCursor, Cursor, CursorMethods},
        traits::{Item, SortedNext},
    },
};
use derive_more::Constructor;
use pyo3::{ffi, prelude::*};
use tap::prelude::*;
pub type IterFwd = Iter<true, Cursor>;
pub type IterRev = Iter<false, Cursor>;
pub type IterBounded = Iter<true, BoundedCursor>;
pub type IterBoundedRev = Iter<false, BoundedCursor>;

#[derive(Debug, Constructor)]
pub struct Iter<const FWD: bool, T> {
    pyref: InnerRef,
    cursor: T,
}

impl<T: ListGetter> From<&Bound<'_, T>> for IterFwd {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        InnerRef::from_owner(owner, owner.get()).into_iterator(Cursor::default())
    }
}
impl<T: ListGetter> From<&Bound<'_, T>> for IterRev {
    #[inline]
    fn from(owner: &Bound<'_, T>) -> Self {
        InnerRef::from_owner(owner, owner.get()).into()
    }
}
impl From<InnerRef> for IterRev {
    #[inline]
    fn from(pyref: InnerRef) -> Self {
        let cursor = pyref.values.len().pipe(Loc::with_pos).into();
        pyref.into_iterator(cursor)
    }
}

impl<const FWD: bool, T: CursorMethods> Iter<FWD, T> {
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
        values
            .get(pos)?
            .get(idx)
            .map(|obj| (at + 1, obj))
            .or_else(|| {
                let p = pos + 1;
                values
                    .get(p)?
                    .first()
                    .map(|obj| (Cursor::new_pos(p) + 1, obj))
            })
    }
    #[inline(always)]
    fn inner_next_back(&self) -> Option<(u64, &Py<PyAny>)> {
        let values = &self.pyref.values;
        let (at, pos, idx) = self.cursor.load();
        values
            .get(pos)
            .and_then(|v| v.get(idx.checked_sub(1)?))
            .map(|obj| (at - 1, obj))
            .or_else(|| {
                let p = pos.checked_sub(1)?;
                let v = values.get(p)?;
                v.last()
                    .map(|obj| (Cursor::new_pos(p) | (v.len() - 1) as u64, obj))
            })
    }
    fn yield_ptr(&self, item: &Py<PyAny>, at: u64) -> *mut ffi::PyObject {
        self.cursor.store(at);
        let ptr = item.as_ptr();
        unsafe { ffi::Py_INCREF(ptr) };
        ptr
    }
}

unsafe impl SortedNext for IterFwd {
    type Rev = Iter<false, Cursor>;
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next().map(|(at, item)| self.yield_ptr(item, at))
    }
    fn rev(&self, py: Python<'_>) -> Self::Rev {
        self.pyref.clone_ref(py).into()
    }
}

unsafe impl SortedNext for IterRev {
    type Rev = Iter<true, Cursor>;
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next_back()
            .map(|(at, item)| self.yield_ptr(item, at))
    }
    fn rev(&self, py: Python<'_>) -> Self::Rev {
        self.pyref.clone_ref(py).into_iterator(Cursor::default())
    }
}

unsafe impl SortedNext for IterBounded {
    type Rev = Iter<false, BoundedCursor>;
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next()
            .filter(|&(at, _)| at <= self.cursor.limit)
            .map(|(at, item)| self.yield_ptr(item, at))
    }
    fn rev(&self, py: Python<'_>) -> Self::Rev {
        self.pyref.clone_ref(py).into_iterator(self.cursor.swap())
    }
}
unsafe impl SortedNext for IterBoundedRev {
    type Rev = Iter<true, BoundedCursor>;
    #[inline(always)]
    unsafe fn next(&self) -> Item {
        self.inner_next_back()
            .filter(|&(at, _)| at >= self.cursor.limit)
            .map(|(at, item)| self.yield_ptr(item, at))
    }
    fn rev(&self, py: Python<'_>) -> Self::Rev {
        self.pyref.clone_ref(py).into_iterator(self.cursor.swap())
    }
}
