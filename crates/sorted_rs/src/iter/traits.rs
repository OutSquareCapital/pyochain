use pyo3::{ffi, prelude::*, types::PyIterator};
use pyo3_ext::{prelude::*, types::PyFrozenClass};
use std::{ops::Deref, ptr};
use tap::Conv;

pub(super) type Item = Option<*mut ffi::PyObject>;

/// Common iterator interface for all sorted iterators.
/// # Safety
/// All implementers must ensure that all pointers are correctly managed,\
/// and that all call sites are restricted to contexts with a valid Python interpreter.
pub unsafe trait SortedNext {
    /// Concrete logic of the python `Iterator`.
    /// # Safety
    /// The caller must ensure that the Iterator is wrapped in a pyclass.
    unsafe fn next(&self) -> Item;
}

/// Trait for the Pyclasses that effectively wrap implementors of `SortedNext`.\
/// You can consider `SortedNext` as the "logic provider" from rust and `PySortedIter` as the "interface provider" for Python.
pub trait PySortedIter:
    PyFrozenClass + Deref<Target = Self::Iter> + Init + From<Self::Iter>
{
    type Iter: SortedNext;
    fn install(py: Python<'_>) {
        unsafe {
            let ty = Self::type_object_raw(py);
            (*ty).tp_iternext = Some(tp_iternext::<Self>);
            ffi::PyType_Modified(ty);
        }
    }
    #[inline(always)]
    #[must_use]
    /// Descriptor to wrap in `__next__` method of the pyclass.\
    /// Note that this will only be called if you do `iter(list).__next__()`, not in hot loops or via `next(it)`.
    fn py_next(slf: Bound<'_, Self>) -> Option<Bound<'_, PyAny>> {
        unsafe {
            slf.cast_into_unchecked::<PyIterator>()
                .next()
                .map(|res| res.unwrap_unchecked())
        }
    }
    /// Full roundtrip from python method call to returned `Iterator`.\
    /// `sorted_collection.method(*args)` -> (bounded collection + bounds, range specs, etc...) -> "pure rust" iterator -> pyclass wrapper -> bounded wrapper.
    #[inline(always)]
    fn build(py: Python<'_>, value: impl Into<Self::Iter>) -> PyResult<Bound<'_, Self>> {
        value.conv::<Self::Iter>().conv::<Self>().into_bound(py)
    }
}

/// The method that is effectively called by Python as the `tp_iternext` body.\
/// We use `Option::unwrap_unchecked` to make `Borrowed::from_ptr_or_opt` act like `Borrowed::from_ptr_unchecked`, which is a private `Pyo3` method.
/// # Safety
/// The caller must ensure that:
/// - the Iterator is wrapped in a valid pyclass
/// - the function is called strictly from a context with a valid Python interpreter, from said pyclass.
#[inline(always)]
unsafe extern "C" fn tp_iternext<T: PySortedIter>(obj: *mut ffi::PyObject) -> *mut ffi::PyObject {
    unsafe {
        let py = Python::assume_attached();
        Borrowed::from_ptr_or_opt(py, obj)
            .unwrap_unchecked()
            .cast_unchecked::<T>()
            .get()
            .next()
            .unwrap_or_else(ptr::null_mut)
    }
}
