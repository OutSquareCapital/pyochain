use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

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

/// Helper to build args tuple: prepend value to args
pub fn build_args<'py>(
    py: Python<'py>,
    value: &Py<PyAny>,
    args: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyTuple>> {
    let mut v: Vec<Bound<'py, PyAny>> = Vec::with_capacity(args.len() + 1);
    v.push(value.bind(py).clone());
    v.extend(args.iter());
    PyTuple::new(py, v)
}

pub fn call_with_self_prepended<'py>(
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    self_ptr: *mut ffi::PyObject,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, pyo3::types::PyDict>>,
) -> PyResult<Py<PyAny>> {
    unsafe {
        let new_argc = args.len() + 1;
        let new_args_ptr = ffi::PyTuple_New(new_argc as ffi::Py_ssize_t);
        if new_args_ptr.is_null() {
            return Err(PyErr::fetch(py));
        }

        ffi::Py_INCREF(self_ptr);
        ffi::PyTuple_SetItem(new_args_ptr, 0, self_ptr);

        // Copy existing args
        for i in 0..args.len() {
            let item = args.get_item(i).unwrap().as_ptr();
            ffi::Py_INCREF(item);
            ffi::PyTuple_SetItem(new_args_ptr, (i + 1) as ffi::Py_ssize_t, item);
        }

        let result = ffi::PyObject_Call(
            func.as_ptr(),
            new_args_ptr,
            kwargs.map(|d| d.as_ptr()).unwrap_or(std::ptr::null_mut()),
        );

        ffi::Py_DECREF(new_args_ptr);

        if result.is_null() {
            Err(PyErr::fetch(py))
        } else {
            Ok(Py::from_owned_ptr(py, result))
        }
    }
}
