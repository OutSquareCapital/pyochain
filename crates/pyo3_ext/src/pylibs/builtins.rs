//! Python `builtins` functions and objects

use crate::pylibs::registery::builtins;
use pyo3::{call::PyCallArgs, ffi, prelude::*, types::PyIterator};
use tap::prelude::*;
/// Create a unique sentinel object. Equivalent to `object()` in Python. On >=3.15, this will become unneded thanks to the new sentinel builtin.
#[inline(always)]
pub fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    builtins::OBJECT.import(py).call0()
}
#[inline(always)]
pub fn max_of<'py, A: PyCallArgs<'py>>(py: Python<'py>, objs: A) -> PyResult<Bound<'py, PyAny>> {
    builtins::MAX.import(py).call1(objs)
}
#[inline(always)]
pub fn min_of<'py, A: PyCallArgs<'py>>(py: Python<'py>, objs: A) -> PyResult<Bound<'py, PyAny>> {
    builtins::MIN.import(py).call1(objs)
}
#[inline(always)]
#[must_use]
pub fn reversed<'py>(sequence: &Bound<'py, PyAny>) -> Bound<'py, PyIterator> {
    unsafe {
        ffi::PyObject_CallOneArg(
            (&raw const ffi::PyReversed_Type)
                .cast::<ffi::PyObject>()
                .cast_mut(),
            sequence.as_ptr(),
        )
        .pipe(|x| Bound::from_owned_ptr(sequence.py(), x))
        .pipe(|x| x.cast_into_unchecked::<PyIterator>())
    }
}
