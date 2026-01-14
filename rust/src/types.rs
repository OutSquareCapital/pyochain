use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyFunction, PyTuple};

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

#[inline]
pub fn call_func<'py>(
    func: &Bound<'py, PyFunction>,
    value: &Bound<'py, PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    match (args.is_empty(), kwargs) {
        (true, None) => func.call1((value,)),
        (true, Some(kw)) => func.call((value,), Some(kw)),
        _ => {
            func.call(
                unsafe {
                    let args_len = args.len();
                    let new_argc = args_len + 1;
                    let new_args_ptr = ffi::PyTuple_New(new_argc as ffi::Py_ssize_t);

                    // PyTuple_SetItem steals the reference, so INCREF first.
                    ffi::Py_INCREF(value.as_ptr());
                    ffi::PyTuple_SetItem(new_args_ptr, 0, value.as_ptr());

                    let args_ptr = args.as_ptr();
                    for i in 0..args_len {
                        let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
                        ffi::Py_INCREF(item);
                        ffi::PyTuple_SetItem(new_args_ptr, (i + 1) as ffi::Py_ssize_t, item);
                    }

                    // Convert owned pointer into Py<PyTuple>
                    Py::<PyTuple>::from_owned_ptr(value.py(), new_args_ptr)
                },
                kwargs,
            )
        }
    }
}
