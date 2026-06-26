mod abc;
mod args;
mod errors;
mod hasher;
mod mixins;
mod option;
mod pylibs;
mod result;
mod tools;
use pyo3::{PyTypeInfo, prelude::*};
use tap::prelude::*;

macro_rules! impl_py_pipe {
    ($type:ty) => {
        #[pymethods]
        impl $type {
            #[pyo3(name = "pipe", signature = (func, *args, **kwargs))]
            fn py_pipe(
                slf: &Bound<'_, Self>,
                func: &Bound<'_, PyAny>,
                args: &args::Args<'_>,
                kwargs: Option<&args::Kwargs<'_>>,
            ) -> PyResult<Py<PyAny>> {
                (
                    args::Concatenate::concat(func, &slf, args, kwargs)?.unbind().pipe(Ok)
                )
            }
        }
    };
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_py_pipe!($first);
        impl_py_pipe!($($rest),+);
    };
}
macro_rules! impl_tap {
    ($type:ty) => {
    #[pymethods]
            impl $type {
    #[pyo3(signature = (f, *args, **kwargs))]
    fn tap(
        slf: &Bound<'_, Self>,
        f: &Bound<'_, PyAny>,
        args: &args::Args<'_>,
        kwargs: Option<&args::Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        args::Concatenate::concat(f, &slf, args, kwargs)?;
        slf.to_owned().into_any().unbind().pipe(Ok)
    }}};
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_tap!($first);
        impl_tap!($($rest),+);
    };
}
impl_tap!(mixins::Fluent, mixins::PyoTap, abc::PyoIterable);
impl_py_pipe!(
    option::PySome,
    option::PyNull,
    result::PyoOk,
    result::PyoErr,
    mixins::Fluent,
    mixins::PyoPipe,
    abc::PyoIterable,
    abc::PyoIterator
);

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
    m.add_class::<mixins::Fluent>()?;
    m.add_class::<mixins::PyoPipe>()?;
    m.add_class::<mixins::PyoTap>()?;
    m.add_wrapped(pyo3::wrap_pymodule!(tools::tools))?;
    m.add_wrapped(pyo3::wrap_pymodule!(abc::abc))?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    sys_mods.set_item("pyochain._tools", m.getattr("_tools")?)?;
    sys_mods.set_item("pyochain.abc._iterator", m.getattr("_iterator")?)?;

    let abc_mod = py.import("collections.abc")?;

    abc_mod
        .getattr("Iterable")?
        .call_method1("register", (abc::PyoIterable::type_object(py),))?;
    abc_mod
        .getattr("Iterator")?
        .call_method1("register", (abc::PyoIterator::type_object(py),))?;
    abc_mod
        .getattr("Container")?
        .call_method1("register", (abc::PyoContainer::type_object(py),))?;
    abc_mod
        .getattr("Sized")?
        .call_method1("register", (abc::PyoSized::type_object(py),))?;

    Ok(())
}
