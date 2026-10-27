use pyo3::{prelude::*, sync::PyOnceLock, types::PyDict};
const SYS: &str = "sys";
static MODULES: PyOnceLock<Py<PyDict>> = PyOnceLock::new();

/// Convenience function to add a module to `sys.modules`. Equivalent to `sys.modules[name] = m`.
pub fn add_module(name: &str, m: &Bound<'_, PyModule>) -> PyResult<()> {
    modules(m.py()).and_then(|modules| modules.set_item(name, m))
}

pub fn modules(py: Python<'_>) -> PyResult<&Bound<'_, PyDict>> {
    MODULES.import(py, SYS, "modules")
}
