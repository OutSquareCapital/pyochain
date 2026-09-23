use crate::abc::{Checkable, PyoIterable};
use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoContainer;

#[pymethods]
impl PyoContainer {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoSized;

#[pymethods]
impl PyoSized {
    #[pyo3(name = "len")]
    fn pyo_len(slf: &Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoCollection;

#[pymethods]
impl PyoCollection {
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
    #[pyo3(name = "len")]
    fn pyo_len(slf: &Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: &Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}
