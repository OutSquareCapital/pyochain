use crate::args::{Args, Concatenate, Kwargs};
use crate::option::{PyNull, PySome};
use crate::result::{PyoErr, PyoOk};
use pyo3::{IntoPyObjectExt, prelude::*};
use tap::prelude::*;

#[pyclass(frozen, subclass, name = "Pipe")]
pub struct PyoPipe;

#[pymethods]
impl PyoPipe {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        PyoPipe {}
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
pub struct Fluent;

#[pymethods]
impl Fluent {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> Self {
        Fluent {}
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
            func.concat(&slf, args, kwargs)?
                .unbind()
                .pipe(PySome::new)
                .into_py_any(py)
        } else {
            PyNull::get(py).into_py_any(py)
        }
    }

    fn then_some(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            slf.to_owned()
                .unbind()
                .into_any()
                .pipe(PySome::new)
                .into_py_any(py)
        } else {
            PyNull::get(py).into_py_any(py)
        }
    }

    fn ok_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(slf
                .to_owned()
                .unbind()
                .into_any()
                .pipe(PyoOk::new)
                .into_py_any(py)?)
        } else {
            Ok(err.to_owned().unbind().pipe(PyoErr::new).into_py_any(py)?)
        }
    }
    fn err_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(slf
                .to_owned()
                .unbind()
                .into_any()
                .pipe(PyoErr::new)
                .into_py_any(py)?)
        } else {
            Ok(err.to_owned().unbind().pipe(PyoOk::new).into_py_any(py)?)
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
            Ok(slf
                .to_owned()
                .unbind()
                .into_any()
                .pipe(PyoOk::new)
                .into_py_any(py)?)
        } else {
            Ok(func
                .concat(&slf, args, kwargs)?
                .unbind()
                .pipe(PyoErr::new)
                .into_py_any(py)?)
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
            Ok(slf
                .to_owned()
                .unbind()
                .into_any()
                .pipe(PyoErr::new)
                .into_py_any(py)?)
        } else {
            Ok(func
                .concat(&slf, args, kwargs)?
                .unbind()
                .pipe(PyoOk::new)
                .into_py_any(py)?)
        }
    }
}
