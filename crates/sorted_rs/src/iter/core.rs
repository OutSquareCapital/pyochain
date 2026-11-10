use std::{ops::Deref, ptr};

use crate::{
    ListGetter,
    inner::InnerList,
    iter::{Iter, cursor::CursorMethods},
};
use derive_more::Constructor;
use pyo3::prelude::*;
use tap::prelude::*;
/// Wrapper type to hold a pointer to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug, Constructor)]
pub(super) struct InnerRef {
    /// Pointer to the `InnerList` of the sorted collection.
    list: *const InnerList,
    /// Keeps the collection alive so `list` stays valid.
    owner: Py<PyAny>,
}
impl Deref for InnerRef {
    type Target = InnerList;
    #[inline(always)]
    fn deref(&self) -> &InnerList {
        unsafe { &*self.list }
    }
}
impl InnerRef {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self::new(self.list, self.owner.clone_ref(py))
    }
    pub fn from_owner<T: ListGetter>(value: &Bound<'_, T>, pyref: &T) -> Self {
        let owner = value.clone().into_any().unbind();
        pyref
            .read()
            .deref()
            .deref()
            .pipe(ptr::from_ref::<InnerList>)
            .pipe(|list| Self::new(list, owner))
    }

    pub fn into_iterator<const FWD: bool, T: CursorMethods>(self, cursor: T) -> Iter<FWD, T> {
        Iter::new(self, cursor)
    }
}
unsafe impl Send for InnerRef {}
unsafe impl Sync for InnerRef {}
