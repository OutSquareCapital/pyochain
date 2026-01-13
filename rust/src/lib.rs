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
    let none_init = PyClassInitializer::from(option::PyochainOption).add_subclass(option::PyNone);
    m.add("NONE", Py::new(py, none_init)?)?;
    m.add_class::<result::PyOk>()?;
    m.add_class::<result::PyErr>()?;
    m.add_class::<types::OptionUnwrapError>()?;
    m.add_class::<types::ResultUnwrapError>()?;
    m.add_class::<result::PyochainResult>()?;
    Ok(())
}
