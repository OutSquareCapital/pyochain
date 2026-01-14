mod converters;
mod option;
mod result;
mod types;
use pyo3::prelude::*;

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<option::PyochainOption>()?;
    m.add_class::<option::PySome>()?;
    m.add_class::<option::PyNone>()?;
    m.add("NONE", option::get_none_singleton(py)?)?;
    m.add_class::<result::PyOk>()?;
    m.add_class::<result::PyErr>()?;
    m.add_class::<option::OptionUnwrapError>()?;
    m.add_class::<result::ResultUnwrapError>()?;
    m.add_class::<result::PyochainResult>()?;
    m.add_class::<converters::Checkable>()?;
    m.add_class::<converters::Pipeable>()?;
    Ok(())
}
