mod args;
mod converters;
mod errors;
mod hasher;
mod option;
mod result;
mod tools;
use pyo3::prelude::*;

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    option::init_null(py)?;
    m.add_class::<option::PyochainOption>()?;
    m.add_class::<option::PySome>()?;
    m.add_class::<option::PyNull>()?;
    m.add_function(wrap_pyfunction!(option::then_if_some, m)?)?;
    m.add_function(wrap_pyfunction!(option::then_if_true, m)?)?;
    m.add_function(wrap_pyfunction!(option::option, m)?)?;
    m.add("NONE", option::get_null(py))?;
    m.add_class::<result::PyOk>()?;
    m.add_class::<result::PyErr>()?;
    m.add_class::<errors::OptionUnwrapError>()?;
    m.add_class::<errors::ResultUnwrapError>()?;
    m.add_class::<result::PyochainResult>()?;
    m.add_class::<converters::Checkable>()?;
    m.add_class::<converters::Pipeable>()?;
    m.add_wrapped(pyo3::wrap_pymodule!(tools::tools))?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyochain._tools", m.getattr("_tools")?)?;

    Ok(())
}
