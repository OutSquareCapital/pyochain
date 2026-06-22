use crate::args::{Args, Kwargs};
use crate::mixins::Checkable;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyIterator;
use tap::prelude::*;

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
    fn iter(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        static ITER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        ITER.import(slf.py(), "pyochain", "Iter")
            .unwrap()
            .call1((slf,))?
            .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
            .pipe(Ok)
    }
}
