use pyo3::{prelude::*, sync::PyOnceLock, types::PyDict};
const SYS: &str = "sys";
static MODULES: PyOnceLock<Py<PyDict>> = PyOnceLock::new();
pub fn modules(py: Python<'_>) -> PyResult<&Bound<'_, PyDict>> {
    MODULES
        .import(py, SYS, "modules")
        .map(|x| unsafe { x.cast_unchecked::<PyDict>() })
}
