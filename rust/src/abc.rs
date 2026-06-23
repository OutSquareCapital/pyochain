use crate::args::{Args, Kwargs};
use crate::mixins::Checkable;
use crate::pylibs;
use pyo3::prelude::*;
use pyo3::types::{PyIterator, PyType};
use tap::prelude::*;

#[pymodule(name = "_abc")]
pub fn abc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyoIterable>()?;
    m.add_class::<PyoIterator>()?;
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
///TODO: once the migration is done, rename it to `PyoIterator`
#[pyclass(name="PyoIteratorRS", subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoIterator;

#[pymethods]
impl PyoIterator {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoIterable {})
            .add_subclass(PyoIterator {})
    }
    #[classmethod]
    fn _from_iterable<'py>(
        cls: Bound<'py, PyType>,
        iterable: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyoIterator>> {
        cls.call1((iterable,))?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyoIterator>() })
            .pipe(Ok)
    }
}
