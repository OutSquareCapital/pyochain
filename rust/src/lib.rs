mod abc;
mod args;
mod errors;
mod hasher;
mod mixins;
mod option;
mod pylibs;
mod result;
mod seq;
mod tools;

use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{PyMapping, PySequence, PyType},
};
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

macro_rules! impl_mapping_view {
    ($type:ty) => {
        #[pymethods]
        impl $type {

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        Ok(format!(
            "{}({:?})",
            slf.get_type().name()?,
            slf.get().mapping.bind(slf.py())
        ))
    }

    fn __len__(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.get().mapping.bind(slf.py()).len()
    }}
    };
    ($first:ty, $($rest:ty),+ $(,)?) => {
        impl_mapping_view!($first);
        impl_mapping_view!($($rest),+);
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
impl_mapping_view!(
    abc::PyoMappingView,
    abc::PyoKeysView,
    abc::PyoValuesView,
    abc::PyoItemsView
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
    m.add_class::<seq::Seq>()?;
    m.add_class::<seq::Range>()?;
    m.add_wrapped(pyo3::wrap_pymodule!(tools_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(abc_mod))?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    sys_mods.set_item("pyochain._tools", m.getattr("_tools")?)?;
    let abc_mod = m.getattr("abc")?;
    sys_mods.set_item("pyochain.abc._iterable", abc_mod.getattr("_iterable")?)?;
    sys_mods.set_item("pyochain.abc._iterator", abc_mod.getattr("_iterator")?)?;
    sys_mods.set_item("pyochain.abc._collection", abc_mod.getattr("_collection")?)?;
    sys_mods.set_item("pyochain.abc._sequences", abc_mod.getattr("_sequences")?)?;
    sys_mods.set_item("pyochain.abc._sets", abc_mod.getattr("_sets")?)?;
    sys_mods.set_item("pyochain.abc._mappings", abc_mod.getattr("_mappings")?)?;
    register_all(py)
}

#[pymodule(name = "_tools")]
pub fn tools_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tools::retain, m)?)?;
    m.add_class::<tools::UniqueIdentity>()?;
    m.add_class::<tools::UniqueKey>()?;
    m.add_class::<tools::Intersperse>()?;
    m.add_class::<tools::MapWindow>()?;
    m.add_class::<tools::MapJuxt>()?;
    m.add_class::<tools::FilterMap>()?;
    m.add_class::<tools::FilterMapStar>()?;
    m.add_class::<tools::Scan>()?;
    m.add_class::<tools::MapWhile>()?;
    m.add_class::<tools::FromFn>()?;
    m.add_class::<tools::Drain>()?;
    m.add_class::<tools::ExtractIf>()?;
    m.add_class::<tools::Successors>()?;
    m.add_class::<tools::FilterStar>()?;
    m.add_class::<tools::WithPosition>()?;
    m.add_class::<tools::ZipLongest>()?;
    m.add_class::<tools::Unzip>()?;
    m.add_class::<tools::GroupBy>()?;
    m.add_class::<tools::Iter>()?;
    m.add_class::<tools::Peekable>()
}
#[pymodule(name = "abc")]
fn abc_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(pyo3::wrap_pymodule!(iterable_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(iterator_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(collection_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(sequences_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(sets_mod))?;
    m.add_wrapped(pyo3::wrap_pymodule!(mappings_mod))
}
#[pymodule(name = "_iterable")]
fn iterable_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoIterable>()
}
#[pymodule(name = "_iterator")]
fn iterator_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoIterator>()
}
#[pymodule(name = "_collection")]
fn collection_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoContainer>()?;
    m.add_class::<abc::PyoSized>()?;
    m.add_class::<abc::PyoCollection>()
}
#[pymodule(name = "_sequences")]
fn sequences_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoReversible>()?;
    m.add_class::<abc::PyoSequence>()?;
    m.add_class::<abc::PyoMutableSequence>()
}
#[pymodule(name = "_sets")]
fn sets_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoSet>()?;
    m.add_class::<abc::PyoMutableSet>()
}
#[pymodule(name = "_mappings")]
fn mappings_mod(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::PyoMappingView>()?;
    m.add_class::<abc::PyoMapping>()?;
    m.add_class::<abc::PyoKeysView>()?;
    m.add_class::<abc::PyoValuesView>()?;
    m.add_class::<abc::PyoItemsView>()?;
    m.add_class::<abc::PyoMutableMapping>()
}

fn register_all(py: Python<'_>) -> PyResult<()> {
    let abc_mod = py.import("collections.abc")?;
    register(&abc_mod, "Iterable", &abc::PyoIterable::type_object(py))?;
    register(&abc_mod, "Iterator", &abc::PyoIterator::type_object(py))?;
    register(&abc_mod, "Container", &abc::PyoContainer::type_object(py))?;
    register(&abc_mod, "Sized", &abc::PyoSized::type_object(py))?;
    register(&abc_mod, "Container", &abc::PyoCollection::type_object(py))?;
    register(&abc_mod, "Sized", &abc::PyoCollection::type_object(py))?;
    register(&abc_mod, "Collection", &abc::PyoCollection::type_object(py))?;
    register(&abc_mod, "Reversible", &abc::PyoReversible::type_object(py))?;
    register(&abc_mod, "Reversible", &abc::PyoSequence::type_object(py))?;
    register(
        &abc_mod,
        "MappingView",
        &abc::PyoMappingView::type_object(py),
    )?;
    register(
        &abc_mod,
        "MutableSequence",
        &abc::PyoMutableSequence::type_object(py),
    )?;
    register(&abc_mod, "Set", &abc::PyoSet::type_object(py))?;
    register(&abc_mod, "MutableSet", &abc::PyoMutableSet::type_object(py))?;
    PySequence::register::<abc::PyoSequence>(py)?;
    register(&abc_mod, "KeysView", &abc::PyoKeysView::type_object(py))?;
    register(&abc_mod, "ValuesView", &abc::PyoValuesView::type_object(py))?;
    register(&abc_mod, "ItemsView", &abc::PyoItemsView::type_object(py))?;
    PyMapping::register::<abc::PyoMapping>(py)?;
    register(
        &abc_mod,
        "MutableMapping",
        &abc::PyoMutableMapping::type_object(py),
    )
}
fn register(abc: &Bound<'_, PyModule>, name: &str, cls: &Bound<'_, PyType>) -> PyResult<()> {
    abc.getattr(name)?
        .call_method1(intern!(abc.py(), "register"), (cls,))?;
    Ok(())
}
