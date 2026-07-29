use crate::mixins::Checkable;
use crate::pyo3_ext::prelude::*;
use crate::tools;
use crate::traits::PyoABC;
use pyo3::prelude::*;
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, tools::Iter>> {
        slf.try_iter().and_then(tools::Iter::new)
    }
}
