use pyo3::prelude::*;

use crate::pylibs::registery::MODULES;

/// Convenience function to add a module to `sys.modules`. Equivalent to `sys.modules[name] = m`.
pub fn add_module(name: &str, m: &Bound<'_, PyModule>) -> PyResult<()> {
    MODULES.import(m.py()).set_item(name, m)
}
