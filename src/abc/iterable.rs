use crate::{abc::Checkable, core::iterators, traits::IntoPyochain};
use pyo3::prelude::*;
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, iterators::Iter>> {
        slf.try_iter()?.into_pyochain()
    }
}
