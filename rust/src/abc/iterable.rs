use crate::abc::Checkable;
use crate::iterators;
use crate::pyo3_ext::prelude::*;
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
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, iterators::Iter>> {
        slf.try_iter().and_then(iterators::Iter::new)
    }
}
