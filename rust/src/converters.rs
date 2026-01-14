use crate::option::{PySome, get_none_singleton};
use crate::result::{PyErr, PyOk};
use crate::types::{PyClassInit, call_func};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
#[pyclass(frozen, subclass)]
pub struct Pipeable;

#[pymethods]
impl Pipeable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Pipeable {}
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &slf, args, kwargs)?.unbind())
    }
    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        slf: &Bound<'_, Self>,
        f: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        call_func(f, &slf, args, kwargs)?;
        Ok(slf.to_owned().into_any().unbind())
    }
}
#[pyclass(frozen, subclass)]
pub struct Checkable;
#[pymethods]
impl Checkable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        Checkable {}
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn then(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();

        if slf.is_truthy()? {
            Ok(PySome::new(call_func(func, &slf, args, kwargs)?.unbind())
                .init(py)?
                .into_any())
        } else {
            get_none_singleton(py)
        }
    }

    fn then_some(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(PySome::new(slf.to_owned().unbind().into_any())
                .init(py)?
                .into_any())
        } else {
            get_none_singleton(py)
        }
    }

    fn ok_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(Py::new(
                py,
                PyOk {
                    value: slf.to_owned().unbind().into_any(),
                },
            )?
            .into_any())
        } else {
            Ok(Py::new(
                py,
                PyErr {
                    error: err.to_owned().unbind(),
                },
            )?
            .into_any())
        }
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn ok_or_else(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.is_truthy()? {
            Ok(Py::new(
                py,
                PyOk {
                    value: slf.to_owned().unbind().into_any(),
                },
            )?
            .into_any())
        } else {
            Ok(Py::new(
                py,
                PyErr {
                    error: call_func(func, &slf, args, kwargs)?.unbind(),
                },
            )?
            .into_any())
        }
    }
}
