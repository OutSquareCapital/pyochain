use pyo3::{prelude::*, sync::PyOnceLock};

const OPERATOR: &str = "operator";
static ITEMGETTER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[inline(always)]
pub fn itemgetter(py: Python<'_>, index: isize) -> PyResult<Bound<'_, PyAny>> {
    ITEMGETTER
        .import(py, OPERATOR, "itemgetter")?
        .call1((index,))
}
