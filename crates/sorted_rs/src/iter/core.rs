use std::{ops::Deref, ptr};

use crate::{ListGetter, inner::InnerData};
use pyo3::prelude::*;
use tap::prelude::*;
/// Wrapper type to hold a pointer to the inner data of a sorted collection.
/// Allow fast iteration without `RwLock` overhead.\
/// This is (as of now) the only way to even COMPETE with sortedcontainers iterator.
#[derive(Debug)]
pub(super) struct InnerRef {
    /// Pointer to the `InnerData` of the sorted collection.
    data: *const InnerData,
    /// Keeps the collection alive so `data` stays valid.
    _owner: Py<PyAny>,
}
impl Deref for InnerRef {
    type Target = InnerData;
    #[inline(always)]
    fn deref(&self) -> &InnerData {
        unsafe { &*self.data }
    }
}
impl InnerRef {
    pub fn new<T: ListGetter>(value: &Bound<'_, T>) -> Self {
        value
            .get()
            .read()
            .deref()
            .deref()
            .pipe(ptr::from_ref::<InnerData>)
            .pipe(|data| Self {
                data,
                _owner: value.clone().into_any().unbind(),
            })
    }
}
unsafe impl Send for InnerRef {}
unsafe impl Sync for InnerRef {}
