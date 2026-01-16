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
/// Convenience helper to call a function with concatenated arguments
pub trait ConcatArgs<'py> {
    fn concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;

    fn concat_star(
        self,
        value: &Bound<'py, PyTuple>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>>;
}
impl<'py> ConcatArgs<'py> for &Bound<'py, PyAny> {
    #[inline]
    fn concat(
        self,
        value: &Bound<'py, PyAny>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match args.is_empty() {
            true => self.call((value,), kwargs),
            _ => self.call(
                unsafe {
                    let args_len = args.len();
                    let new_argc = args_len + 1;
                    let new_args_ptr = ffi::PyTuple_New(new_argc as ffi::Py_ssize_t);
                    ffi::Py_INCREF(value.as_ptr());
                    ffi::PyTuple_SetItem(new_args_ptr, 0, value.as_ptr());

                    let args_ptr = args.as_ptr();
                    for i in 0..args_len {
                        let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
                        ffi::Py_INCREF(item);
                        ffi::PyTuple_SetItem(new_args_ptr, (i + 1) as ffi::Py_ssize_t, item);
                    }
                    Py::<PyTuple>::from_owned_ptr(value.py(), new_args_ptr)
                },
                kwargs,
            ),
        }
    }

    #[inline]
    fn concat_star(
        self,
        value: &Bound<'py, PyTuple>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match args.is_empty() {
            true => self.call(value, kwargs),
            _ => self.call(
                unsafe {
                    let tuple_len = value.len();
                    let args_len = args.len();
                    let total_len = tuple_len + args_len;
                    let new_args_ptr = ffi::PyTuple_New(total_len as ffi::Py_ssize_t);
                    let tuple_ptr = value.as_ptr();
                    for i in 0..tuple_len {
                        let item = ffi::PyTuple_GET_ITEM(tuple_ptr, i as ffi::Py_ssize_t);
                        ffi::Py_INCREF(item);
                        ffi::PyTuple_SetItem(new_args_ptr, i as ffi::Py_ssize_t, item);
                    }
                    let args_ptr = args.as_ptr();
                    for i in 0..args_len {
                        let item = ffi::PyTuple_GET_ITEM(args_ptr, i as ffi::Py_ssize_t);
                        ffi::Py_INCREF(item);
                        ffi::PyTuple_SetItem(
                            new_args_ptr,
                            (tuple_len + i) as ffi::Py_ssize_t,
                            item,
                        );
                    }

                    Py::<PyTuple>::from_owned_ptr(value.py(), new_args_ptr)
                },
                kwargs,
            ),
        }
    }
}
