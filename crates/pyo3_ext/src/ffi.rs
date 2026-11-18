use std::ffi::c_int;

use pyo3::ffi;

unsafe extern "C" {
    pub fn PyUnstable_Long_IsCompact(op: *const ffi::PyLongObject) -> c_int;
    pub fn PyUnstable_Long_CompactValue(op: *const ffi::PyLongObject) -> ffi::Py_ssize_t;
}
