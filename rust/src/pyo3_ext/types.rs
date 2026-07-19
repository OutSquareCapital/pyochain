use crate::pyobject_native_type_named;
use pyo3::ffi::PyTypeObject;
/// This module contains Python built-in functions and objects, as well as functions and objects from the `itertools` and `functools` modules.
/// Each submodule declares a const string with the name of the module, and a const `PyOnceLock` + associated fn for each function or object that is imported from that module.
/// This pattern ensure maximum performance by only importing the function or object once, and reusing it for subsequent calls.
/// We also use unsafe casts to correct types, aggressive inlining, and `&Bound` to maximize performance.
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyFrozenSet, PyInt, PyList, PySequence, PySet, PySlice, PyType};
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
pub struct PySupportsIndex(PyAny);
pyobject_native_type_named!(PySupportsIndex);
unsafe impl PyTypeInfo for PySupportsIndex {
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
pub trait PySupportsIndexMethods<'py> {
    fn index(&self) -> PyResult<Bound<'py, PyInt>>;
}
impl<'py> PySupportsIndexMethods<'py> for Bound<'py, PySupportsIndex> {
    fn index(&self) -> PyResult<Bound<'py, PyInt>> {
        self.call_method0(intern!(self.py(), "__index__"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
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
/// Type representing the `collections.deque` class.
#[repr(transparent)]
pub struct PyDeque(PyAny);
pyobject_native_type_named!(PyDeque);
unsafe impl PyTypeInfo for PyDeque {
    const NAME: &'static str = "deque";
    const MODULE: Option<&'static str> = Some("collections");

    #[inline]
    fn type_object_raw(py: Python<'_>) -> *mut PyTypeObject {
        static TYPE: PyOnceLock<Py<PyType>> = PyOnceLock::new();
        TYPE.import(py, "collections", "deque")
            .unwrap()
            .as_type_ptr()
    }

    #[inline]
    fn is_type_of(object: &Bound<'_, PyAny>) -> bool {
        object
            .is_instance(&Self::type_object(object.py()).into_any())
            .unwrap_or_else(|err| {
                err.write_unraisable(object.py(), Some(object));
                false
            })
    }
}

impl PyDeque {
    #[inline]
    pub fn new<'py>(
        py: Python<'py>,
        iterable: Bound<'py, PyAny>,
        maxlen: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::type_object(py)
            .call1((iterable, maxlen))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}
pub trait PyDequeMethods<'py> {
    /// Returns `self` cast as a `PySequence`.
    fn as_sequence(&self) -> &Bound<'py, PySequence>;
    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()>;
    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()>;
    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()>;
    fn clear(&self) -> PyResult<()>;
    fn extend_left(&self, iterable: Bound<'_, PyAny>) -> PyResult<()>;
    fn rotate(&self, n: isize) -> PyResult<()>;
    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()>;
    fn count(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>>;
}
impl<'py> PyDequeMethods<'py> for Bound<'py, PyDeque> {
    /// Returns `self` cast as a `PySequence`.
    fn as_sequence(&self) -> &Bound<'py, PySequence> {
        unsafe { self.cast_unchecked() }
    }

    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "append"), (x,))?;
        Ok(())
    }

    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "appendleft"), (x,))?;
        Ok(())
    }

    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "extend"), (iterable,))?;
        Ok(())
    }

    fn clear(&self) -> PyResult<()> {
        self.call_method0(intern!(self.py(), "clear"))?;
        Ok(())
    }

    fn extend_left(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "extendleft"), (iterable,))?;
        Ok(())
    }

    fn rotate(&self, n: isize) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "rotate"), (n,))?;
        Ok(())
    }

    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.call_method1(intern!(self.py(), "insert"), (index, value))?;
        Ok(())
    }
    fn count(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.call_method1(intern!(self.py(), "count"), (value,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
    }
}
