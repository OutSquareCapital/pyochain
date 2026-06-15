use crate::args::{Args, Concatenate, Kwargs};
use crate::option::{PyNull, PySome};
use crate::result::{PyoErr, PyoOk};
use pyo3::{IntoPyObjectExt, prelude::*};

#[pyclass(frozen, subclass, name = "Into")]
pub struct PyoInto;

#[pymethods]
impl PyoInto {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        PyoInto {}
    }
}

#[pyclass(frozen, subclass, name = "Tap")]
pub struct PyoTap;
#[pymethods]
impl PyoTap {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        PyoTap {}
    }
}
#[pyclass(frozen, subclass)]
pub struct Pipeable;

#[pymethods]
impl Pipeable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        Pipeable {}
    }
}
#[pyclass(frozen, subclass)]
pub struct Checkable;
#[pymethods]
impl Checkable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        Checkable {}
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn then(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();

        if slf.is_truthy()? {
            PySome::new(func.concat(&slf, args, kwargs)?.unbind()).into_py_any(py)
        } else {
            PyNull::get(py).into_py_any(py)
        }
    }

    fn then_some(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            PySome::new(slf.to_owned().unbind().into_any()).into_py_any(py)
        } else {
            PyNull::get(py).into_py_any(py)
        }
    }

    fn ok_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(PyoOk::new(slf.to_owned().unbind().into_any()).into_py_any(py)?)
        } else {
            Ok(PyoErr::new(err.to_owned().unbind()).into_py_any(py)?)
        }
    }
    fn err_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(PyoErr::new(slf.to_owned().unbind().into_any()).into_py_any(py)?)
        } else {
            Ok(PyoOk::new(err.to_owned().unbind()).into_py_any(py)?)
        }
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn ok_or_else(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(PyoOk::new(slf.to_owned().unbind().into_any()).into_py_any(py)?)
        } else {
            Ok(PyoErr::new(func.concat(&slf, args, kwargs)?.unbind()).into_py_any(py)?)
        }
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn err_or_else(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(PyoErr::new(slf.to_owned().unbind().into_any()).into_py_any(py)?)
        } else {
            Ok(PyoOk::new(func.concat(&slf, args, kwargs)?.unbind()).into_py_any(py)?)
        }
    }
}
