use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Exception raised when unwrapping fails on Result types
#[pyclass(extends = PyValueError)]
pub struct ResultUnwrapError;
#[pymethods]
impl ResultUnwrapError {
    #[new]
    fn new(_exc_arg: &Bound<'_, PyAny>) -> Self {
        ResultUnwrapError
    }
}

/// Exception raised when unwrapping fails on Option types
#[pyclass(extends = PyValueError)]
pub struct OptionUnwrapError;

#[pymethods]
impl OptionUnwrapError {
    #[new]
    fn new(_exc_arg: &Bound<'_, PyAny>) -> Self {
        OptionUnwrapError
    }
}
