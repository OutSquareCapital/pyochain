use pyo3::{
    ffi,
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
    #[inline(always)]
    fn is_compact(&self) -> bool {
        unsafe { (*self.as_ptr().cast::<LongObject>()).is_valid() }
    }
    #[inline(always)]
    unsafe fn compact_value(&self) -> isize {
        debug_assert!(self.is_compact());
        unsafe { (*self.as_ptr().cast::<LongObject>()).compact_value() }
    }
}
/// Layout of `PyLongObject` since `CPython` 3.12 (`_PyLongValue` inlined), which is the minimum supported version.\
/// `pyo3-ffi` only exposes it as an opaque struct, and `PyUnstable_Long_*` aren't inlinable across the DLL boundary.
#[repr(C)]
pub struct LongObject {
    ob_base: ffi::PyObject,
    lv_tag: usize,
    ob_digit: [u32; 1],
}
impl LongObject {
    /// `lv_tag` of a compact int is < `2 << _PyLong_NON_SIZE_BITS` (3).
    const COMPACT_TAG_LIMIT: usize = 16;
    /// Sign is stored in the low 2 bits of `lv_tag`: 0 = positive, 1 = zero, 2 = negative.
    const SIGN_MASK: usize = 3;
    pub fn is_valid(&self) -> bool {
        self.lv_tag < Self::COMPACT_TAG_LIMIT
    }
    pub fn compact_value(&self) -> isize {
        let sign = 1 - (self.lv_tag & Self::SIGN_MASK).cast_signed();
        sign * (self.ob_digit[0] as usize).cast_signed()
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
