use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
};

#[inline]
pub fn not_in_list_err<T>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let msg = format!("{} not in list", value.repr()?);
    Err(PyValueError::new_err(msg))
}
#[inline]
pub fn is_not_in_list_err<T>(value: &Bound<'_, PyAny>) -> PyResult<T> {
    let msg = format!("{} is not in list", value.repr()?);
    Err(PyValueError::new_err(msg))
}
#[inline]
pub fn out_of_range_err<T>() -> PyResult<T> {
    let msg = "list index out of range";
    Err(PyIndexError::new_err(msg))
}
