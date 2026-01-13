use pyo3::exceptions::PyValueError;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

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
    py: Python<'py>,
    func: &Bound<'py, PyAny>,
    value: &Py<PyAny>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    match (args.is_empty(), kwargs) {
        (true, None) => {
            // Fast path: single arg, no kwargs - use call1
            func.call1((&value,))
        }
        (true, Some(kw)) => {
            // Fast path: single arg with kwargs
            func.call((&value,), Some(kw))
        }
        _ => {
            // Slow path: concatenate
            func.call(&unsafe { concatenate(py, value, args) }, kwargs)
        }
    }
}
/// Helper to build args tuple: prepend value to args
/// This is equivalent to `Concatenate[T, P]` in Python typing.
/// Safety: caller must ensure that `args` contains valid borrowed references
/// and that the generated tuple allocation succeeds. This function will
/// create a new `PyTuple` via the raw C API and return an owned `Py<PyTuple>`.
#[inline]
pub unsafe fn concatenate<'py>(
    py: Python<'py>,
    value: &Py<PyAny>,
    args: &Bound<'py, PyTuple>,
) -> Py<PyTuple> {
    unsafe {
        let args_len = args.len();
        let new_argc = args_len + 1;
        let new_args_ptr = ffi::PyTuple_New(new_argc as ffi::Py_ssize_t);

        // Note: we intentionally omit null checks here for maximum speed; caller
        // must guarantee the interpreter state and allocation succeeded.
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
        Py::<PyTuple>::from_owned_ptr(py, new_args_ptr)
    }
}
