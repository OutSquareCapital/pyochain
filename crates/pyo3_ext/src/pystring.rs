use std::hint::unreachable_unchecked;

use pyo3::{ffi, prelude::*, types::PyString};
use tap::Pipe;
pub trait PyStringExtMethods {
    /// Number of code points. <https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_GET_LENGTH>
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// <https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_KIND>
    fn kind(&self) -> UnicodeByteKind;
    /// <https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_1BYTE_DATA>
    /// # Safety
    /// The string must be of kind `UnicodeByteKind::One`, otherwise the returned value is meaningless.
    unsafe fn as_bytes_8(&self) -> &[u8];
}
impl PyStringExtMethods for Bound<'_, PyString> {
    fn len(&self) -> usize {
        unsafe { ffi::PyUnicode_GET_LENGTH(self.as_ptr()) }.cast_unsigned()
    }
    fn kind(&self) -> UnicodeByteKind {
        match unsafe { ffi::PyUnicode_KIND(self.as_ptr()) } {
            ffi::PyUnicode_1BYTE_KIND => UnicodeByteKind::One,
            ffi::PyUnicode_2BYTE_KIND => UnicodeByteKind::Two,
            ffi::PyUnicode_4BYTE_KIND => UnicodeByteKind::Four,
            _ => unsafe { unreachable_unchecked() },
        }
    }
    unsafe fn as_bytes_8(&self) -> &[u8] {
        debug_assert!(matches!(self.kind(), UnicodeByteKind::One));
        unsafe {
            self.as_ptr()
                .pipe(|ptr| ffi::PyUnicode_DATA(ptr))
                .cast::<u8>()
                .pipe(|data| std::slice::from_raw_parts(data, self.len()))
        }
    }
}
/// <https://docs.python.org/3/c-api/unicode.html#c.PyUnicode_1BYTE_DATA>
pub enum UnicodeByteKind {
    One = 1,
    Two = 2,
    Four = 4,
}
