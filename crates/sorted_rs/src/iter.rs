use crate::{Bounds, bounds::AtomicBounds, inner::InnerData};
use derive_more::Constructor;
use parking_lot::RwLock;
use pyo3::ffi;
use std::{ops::Deref, sync::Arc};

#[derive(Constructor, Debug)]
pub struct Iter<T> {
    _owner: Arc<RwLock<T>>,
    data: *const T,
    bounds: AtomicBounds,
    reversed: bool,
}

impl<T: Deref<Target = InnerData>> Iter<T> {
    pub fn bounded(owner: Arc<RwLock<T>>, bounds: AtomicBounds, reversed: bool) -> Self {
        let data = {
            let guard = owner.read();
            std::ptr::from_ref::<T>(&*guard)
        };

        Self::new(owner, data, bounds, reversed)
    }
    pub fn full(owner: Arc<RwLock<T>>, reversed: bool) -> Self {
        let guard = owner.read();
        let bounds = Bounds::from_full_iter(&guard);
        let data = std::ptr::from_ref::<T>(&*guard);
        drop(guard);
        Self::new(owner, data, bounds.into(), reversed)
    }

    unsafe fn get_data(&self) -> &T {
        unsafe { &*self.data }
    }
    /// # Safety
    /// This function is NOT supposed to be called from Rust.
    #[inline(always)]
    pub unsafe fn next(&self) -> *mut ffi::PyObject {
        let data = unsafe { self.get_data() };

        let (min_pos, min_idx) = self.bounds.min.load();
        let (max_pos, max_idx) = self.bounds.max.load();

        if min_pos == max_pos && min_idx == max_idx {
            std::ptr::null_mut()
        } else if self.reversed {
            let (new_pos, new_idx) = if max_idx > 0 {
                (max_pos, max_idx - 1)
            } else {
                let pos = max_pos - 1;
                let v = unsafe { data.values.get_unchecked(pos) };
                (pos, v.len() - 1)
            };

            let item = unsafe { data.values.get_unchecked(new_pos).get_unchecked(new_idx) };

            let ptr = item.as_ptr();

            unsafe {
                ffi::Py_INCREF(ptr);
            }
            self.bounds.max.store(new_pos, new_idx);

            ptr
        } else {
            let v = unsafe { data.values.get_unchecked(min_pos) };

            let item = unsafe { v.get_unchecked(min_idx) };
            let ptr = item.as_ptr();

            unsafe {
                ffi::Py_INCREF(ptr);
            }

            let (new_pos, new_idx) = if min_pos + 1 < data.values.len() && min_idx + 1 >= v.len() {
                (min_pos + 1, 0)
            } else {
                (min_pos, min_idx + 1)
            };

            self.bounds.min.store(new_pos, new_idx);

            ptr
        }
    }
}

unsafe impl<T: Send + Sync> Send for Iter<T> {}
unsafe impl<T: Send + Sync> Sync for Iter<T> {}
