use crate::{abc::Checkable, core::iterators, traits::IntoPyochain};
use pyo3::prelude::*;
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    fn iter(slf: Bound<'_, Self>) -> PyResult<Bound<'_, iterators::Iter>> {
        slf.try_iter()?.into_pyochain()
    }
}
