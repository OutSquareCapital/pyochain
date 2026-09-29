use crate::abc;
use derive_more::From;
use pyo3::{PyTypeInfo, ffi, prelude::*};
use sorted_rs::iter;
macro_rules! impl_sorted_iter {
    ($($iter:ty => $pyiter:ident),+ $(,)?) => {
        $(
            #[derive(From)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $pyiter($iter);
            impl $pyiter {
                #[inline(always)]
                unsafe extern "C" fn tp_iternext(
                    obj: *mut pyo3::ffi::PyObject,
                ) -> *mut pyo3::ffi::PyObject {
                    let py = unsafe { Python::assume_attached() };
                    unsafe { Borrowed::from_ptr(py, obj).cast_unchecked::<Self>().get().0.next() }
                }
                pub fn install_iternext(py: Python<'_>) {
                    unsafe {
                        let ty = Self::type_object_raw(py);
                        (*ty).tp_iternext = Some(Self::tp_iternext);
                        ffi::PyType_Modified(ty);
                    }
                }
            }
        )+
    };
}

impl_sorted_iter!(
    iter::Iter => SortedIter,
    iter::IterRev => SortedIterRev,
    iter::IterBounded => SortedIterBounded,
    iter::IterBoundedRev => SortedIterBoundedRev,
);
