use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
    types::PyString,
};

#[inline]
pub fn not_in_list(repr: &Bound<'_, PyString>) -> PyErr {
    let msg = format!("{repr} not in list");
    PyValueError::new_err(msg)
}
#[inline]
pub fn out_of_range() -> PyErr {
    let msg = "list index out of range";
    PyIndexError::new_err(msg)
}
