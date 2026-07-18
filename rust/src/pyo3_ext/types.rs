use crate::pyobject_native_type_named;
use pyo3::ffi::PyTypeObject;
/// This module contains Python built-in functions and objects, as well as functions and objects from the `itertools` and `functools` modules.
/// Each submodule declares a const string with the name of the module, and a const `PyOnceLock` + associated fn for each function or object that is imported from that module.
/// This pattern ensure maximum performance by only importing the function or object once, and reusing it for subsequent calls.
/// We also use unsafe casts to correct types, aggressive inlining, and `&Bound` to maximize performance.
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyList, PySet, PySlice, PyType};
use pyo3::{PyTypeInfo, intern, prelude::*};

/// All ABCs have a `register` method that can be used to register a type as a virtual subclass of the ABC.\
/// This trait factorize the implementation for all ABCs.\
/// The code is strictly identical from what's already available for `pyo3::types::PySequence` for example.
pub trait ABCRegister<'py>: PyTypeInfo {
    fn register<T: PyTypeInfo>(py: Python<'_>) -> PyResult<()> {
        let ty = T::type_object(py);
        Self::type_object(py).call_method1("register", (ty,))?;
        Ok(())
    }
}

impl ABCRegister<'_> for PyMutableSequence {}
impl ABCRegister<'_> for PyAbstractSet {}

/// Type representing the `typing.SupportsIndex` protocol.
#[repr(transparent)]
pub struct SupportsIndex(PyAny);
pyobject_native_type_named!(SupportsIndex);
unsafe impl PyTypeInfo for SupportsIndex {
    const NAME: &'static str = "SupportsIndex";
    const MODULE: Option<&'static str> = Some("typing");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "typing", "SupportsIndex")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        let py = object.py();
        object
            .hasattr(intern!(py, "__index__"))
            .unwrap_or_else(|err| {
                err.write_unraisable(object.py(), Some(object));
                false
            })
            || object
                .is_instance(&Self::type_object(py).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(py, Some(object));
                    false
                })
    }
}
/// Type representing the `collections.abc.MutableSequence` abstract base class.
#[repr(transparent)]
pub struct PyMutableSequence(PyAny);
pyobject_native_type_named!(PyMutableSequence);
unsafe impl PyTypeInfo for PyMutableSequence {
    const NAME: &'static str = "MutableSequence";
    const MODULE: Option<&'static str> = Some("collections.abc");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "collections.abc", "MutableSequence")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PyList::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}
pub trait PyMutableSequenceMethods<'py> {
    fn set_slice_with_step(
        &self,
        start: isize,
        stop: isize,
        step: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()>;
}
impl<'py> PyMutableSequenceMethods<'py> for Bound<'py, PyMutableSequence> {
    fn set_slice_with_step(
        &self,
        start: isize,
        stop: isize,
        step: isize,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = value.py();

        let slice = PySlice::new(py, start, stop, step);
        self.set_item(&slice, value)
    }
}
/// Type representing the `collections.abc.Set` abstract base class.
#[repr(transparent)]
pub struct PyAbstractSet(PyAny);
pyobject_native_type_named!(PyAbstractSet);
unsafe impl PyTypeInfo for PyAbstractSet {
    const NAME: &'static str = "Set";
    const MODULE: Option<&'static str> = Some("collections.abc");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "collections.abc", "Set")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        PySet::is_type_of(object)
            || PyFrozenSet::is_type_of(object)
            || object
                .is_instance(&Self::type_object(object.py()).into_any())
                .unwrap_or_else(|err| {
                    err.write_unraisable(object.py(), Some(object));
                    false
                })
    }
}
