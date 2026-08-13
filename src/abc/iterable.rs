use crate::{
    abc::Checkable,
    core::iterators,
    traits::{IntoPyochain, PyoABC},
};
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, iterators::Iter>> {
        slf.try_iter()?.into_pyochain()
    }
}
