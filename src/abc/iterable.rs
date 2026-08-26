use crate::{abc::Checkable, core::iterators};
use pyo3::prelude::*;
use pyo3_ext::prelude::*;
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    fn iter<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, iterators::Iter>> {
        slf.try_iter()?.try_into_py()
    }
}
