use std::{ops::Deref, ptr};

use crate::{
    ListGetter,
    inner::InnerData,
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
    /// Pointer to the `InnerData` of the sorted collection.
    data: *const InnerData,
    /// Keeps the collection alive so `data` stays valid.
    owner: Py<PyAny>,
}
impl Deref for InnerRef {
    type Target = InnerData;
    #[inline(always)]
    fn deref(&self) -> &InnerData {
        unsafe { &*self.data }
    }
}
impl InnerRef {
    pub fn clone_ref(&self, py: Python<'_>) -> Self {
        Self::new(self.data, self.owner.clone_ref(py))
    }
    pub fn from_owner<T: ListGetter>(value: &Bound<'_, T>, pyref: &T) -> Self {
        let owner = value.clone().into_any().unbind();
        pyref
            .read()
            .deref()
            .deref()
            .pipe(ptr::from_ref::<InnerData>)
            .pipe(|data| Self::new(data, owner))
    }

    pub fn into_iterator<const FWD: bool, T: CursorMethods>(self, cursor: T) -> Iter<FWD, T> {
        Iter::new(self, cursor)
    }
}
unsafe impl Send for InnerRef {}
unsafe impl Sync for InnerRef {}
