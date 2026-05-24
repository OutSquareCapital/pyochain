mod args;
mod errors;
mod hasher;
mod mixins;
mod option;
mod result;
mod tools;
use pyo3::prelude::*;

macro_rules! impl_py_into {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            #[pyo3(name = "into", signature = (func, *args, **kwargs))]
            fn py_into(
                slf: &Bound<'_, Self>,
                func: &Bound<'_, PyAny>,
                args: &args::Args<'_>,
                kwargs: Option<&args::Kwargs<'_>>,
            ) -> PyResult<Py<PyAny>> {
                Ok(
                    args::Concatenate::concat(func, &slf, args, kwargs)?.unbind(),
                )
            }
        }
    };
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_py_into!($first);
        impl_py_into!($($rest),+);
    };
}
macro_rules! impl_inspect {
    ($type:ty) => {
    #[pymethods]
            impl $type {
    #[pyo3(signature = (f, *args, **kwargs))]
    fn inspect(
        slf: &Bound<'_, Self>,
        f: &Bound<'_, PyAny>,
        args: &args::Args<'_>,
        kwargs: Option<&args::Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        args::Concatenate::concat(f, &slf, args, kwargs)?;
        Ok(slf.to_owned().into_any().unbind())
    }}};
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_inspect!($first);
        impl_inspect!($($rest),+);
    };
}
impl_inspect!(mixins::Pipeable, mixins::PyoInspect);
impl_py_into!(option::PySome, option::PyNull, result::PyoOk, result::PyoErr, mixins::Pipeable, mixins::PyoInto);

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    option::PyNull::init(py)?;
    m.add_class::<option::PyochainOption>()?;
    m.add_class::<option::PySome>()?;
    m.add_class::<option::PyNull>()?;
    m.add_function(wrap_pyfunction!(option::then_if_some, m)?)?;
    m.add_function(wrap_pyfunction!(option::then_if_true, m)?)?;
    m.add_function(wrap_pyfunction!(option::option, m)?)?;
    m.add("NONE", option::PyNull::get(py))?;
    m.add_class::<result::PyoOk>()?;
    m.add_class::<result::PyoErr>()?;
    m.add_class::<errors::OptionUnwrapError>()?;
    m.add_class::<errors::ResultUnwrapError>()?;
    m.add_class::<result::PyochainResult>()?;
    m.add_class::<mixins::Checkable>()?;
    m.add_class::<mixins::Pipeable>()?;
    m.add_class::<mixins::PyoInto>()?;
    m.add_class::<mixins::PyoInspect>()?;
    m.add_wrapped(pyo3::wrap_pymodule!(tools::tools))?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("pyochain._tools", m.getattr("_tools")?)?;

    Ok(())
}
