use crate::args::{Args, Kwargs};
use crate::mixins::Checkable;
use crate::pylibs;
use pyo3::prelude::*;
use pyo3::types::PyIterator;

#[pymodule(name = "_abc")]
pub fn abc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyoIterable>()?;
    Ok(())
}
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(PyoIterable {})
    }
    fn iter<'py>(slf: &'py Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        pylibs::pyochain::iter::new(slf)
    }
}
