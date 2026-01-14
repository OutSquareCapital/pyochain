use crate::option::{PySome, PyochainOption, get_none_singleton};
use crate::result::{PyErr, PyOk};
use crate::types::call_func;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFunction, PyTuple};
#[pyclass(frozen, subclass)]
pub struct Pipeable;

#[pymethods]
impl Pipeable {
    #[new]
    fn new() -> Self {
        Pipeable {}
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn into(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyFunction>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(call_func(func, &slf, args, kwargs)?.unbind())
    }
    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        slf: &Bound<'_, Self>,
        f: &Bound<'_, PyFunction>,
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
    fn new() -> Self {
        Checkable {}
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn then(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyFunction>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.call_method0("__bool__")?.unbind().is_truthy(py)? {
            get_none_singleton(py)
        } else {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
                value: call_func(func, &slf, args, kwargs)?.unbind(),
            });
            Ok(Py::new(py, init)?.into_any())
        }
    }

    fn then_some(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.call_method0("__bool__")?.unbind().is_truthy(py)? {
            let init = PyClassInitializer::from(PyochainOption).add_subclass(PySome {
                value: slf.to_owned().unbind().into_any(),
            });
            Ok(Py::new(py, init)?.into_any())
        } else {
            get_none_singleton(py)
        }
    }

    fn ok_or(slf: &Bound<'_, Self>, err: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.call_method0("__bool__")?.unbind().is_truthy(py)? {
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
        func: &Bound<'_, PyFunction>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        if slf.call_method0("__bool__")?.unbind().is_truthy(py)? {
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
