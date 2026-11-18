use pyo3::{
    exceptions::{PyIndexError, PyValueError},
    prelude::*,
};

#[inline]
pub fn not_in_list(value: &Bound<'_, PyAny>) -> PyErr {
    match value.repr() {
        Ok(repr) => {
            let msg = format!("{repr} not in list");
            PyValueError::new_err(msg)
        }
        Err(err) => err,
    }
}
#[inline]
pub fn out_of_range() -> PyErr {
    let msg = "list index out of range";
    PyIndexError::new_err(msg)
}
