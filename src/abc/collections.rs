use crate::{
    abc::{Checkable, PyoIterable},
    iterators,
    traits::PyoABC,
};
use pyo3::prelude::*;
use pyo3_ext::{
    args::{Args, Kwargs},
    pylibs,
};
use tap::Pipe;

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoContainer;

#[pymethods]
impl PyoContainer {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoSized;

#[pymethods]
impl PyoSized {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
    #[pyo3(name = "len")]
    fn pyo_len(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoCollection;

#[pymethods]
impl PyoCollection {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
    #[pyo3(name = "len")]
    fn pyo_len(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoReversible;

#[pymethods]
impl PyoReversible {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }
    /// We use unsafe code here because calling `reversed` with `PyOnceLock` pattern is 2x slower than pure python for some reason.
    fn rev(slf: Bound<'_, Self>) -> PyResult<Bound<'_, iterators::Iter>> {
        slf.as_any()
            .pipe(pylibs::builtins::reversed)
            .pipe(iterators::Iter::new)
    }
}
