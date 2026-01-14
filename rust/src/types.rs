use pyo3::PyClass;

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

// Convenience helper to avoid nested calls
pub trait PyClassInit {
    type Class: PyClass;
    /// Creates a new instance Py<T> of a #[pyclass] on the Python heap.
    fn init(self, py: Python<'_>) -> PyResult<Py<Self::Class>>;
}

impl<T: PyClass> PyClassInit for PyClassInitializer<T> {
    type Class = T;
    #[inline]
    fn init(self, py: Python<'_>) -> PyResult<Py<T>> {
        Py::new(py, self)
    }
}

#[inline]
pub fn call_func<'py>(
    func: &Bound<'py, PyAny>,
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
