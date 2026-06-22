use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::PyFunction;
use tap::prelude::*;
#[inline(always)]
fn itertools(py: Python<'_>) -> Bound<'_, PyModule> {
    PyModule::import(py, intern!(py, "itertools")).unwrap()
}
#[inline(always)]
pub fn tee(py: Python<'_>) -> Bound<'_, PyFunction> {
    itertools(py)
        .getattr(intern!(py, "tee"))
        .unwrap()
        .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyFunction>() })
}
#[inline(always)]
pub fn group_by(py: Python<'_>) -> Bound<'_, PyFunction> {
    itertools(py)
        .getattr(intern!(py, "groupby"))
        .unwrap()
        .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyFunction>() })
}
