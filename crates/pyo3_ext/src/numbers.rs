use pyo3::{
    prelude::*,
    types::{PyFloat, PyInt},
};
pub trait PyIntExtMethods {
    /// <https://docs.python.org/3/c-api/long.html#c.PyUnstable_Long_IsCompact>
    fn is_compact(&self) -> bool;
    /// <https://docs.python.org/3/c-api/long.html#c.PyUnstable_Long_CompactValue>
    ///
    /// # Safety
    /// [`is_compact`](Self::is_compact) must be `true`, otherwise the returned value is meaningless.
    unsafe fn compact_value(&self) -> isize;
}
impl PyIntExtMethods for Bound<'_, PyInt> {
    fn is_compact(&self) -> bool {
        unsafe { crate::ffi::PyUnstable_Long_IsCompact(self.as_ptr().cast()) != 0 }
    }
    unsafe fn compact_value(&self) -> isize {
        debug_assert!(self.is_compact());
        unsafe { crate::ffi::PyUnstable_Long_CompactValue(self.as_ptr().cast()) }
    }
}
pub trait PyFloatExtMethods {
    /// <https://docs.python.org/3/c-api/float.html#c.PyFloat_AS_DOUBLE>
    fn as_double(&self) -> f64;
}
impl PyFloatExtMethods for Bound<'_, PyFloat> {
    fn as_double(&self) -> f64 {
        unsafe { pyo3::ffi::PyFloat_AS_DOUBLE(self.as_ptr()) }
    }
}
