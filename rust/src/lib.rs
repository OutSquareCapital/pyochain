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
            slf.get()._mapping.bind(slf.py())
        ))
    }

    fn __len__(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.get()._mapping.bind(slf.py()).len()
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

    let range_mod = PyModule::new(py, "_range")?;
    m.add_submodule(&range_mod)?;
    range_mod.add_class::<seq::Range>()?;
    let seq_mod = PyModule::new(py, "_seq")?;
    m.add_submodule(&seq_mod)?;
    seq_mod.add_class::<seq::Seq>()?;
    let vec_mod = PyModule::new(py, "_vec")?;
    m.add_submodule(&vec_mod)?;
    vec_mod.add_class::<seq::Vec>()?;
    let set_mod = PyModule::new(py, "_set")?;
    m.add_submodule(&set_mod)?;
    set_mod.add_class::<seq::Set>()?;
    set_mod.add_class::<seq::SetMut>()?;
    let tools_mod = PyModule::new(py, "_tools")?;

    tools_mod.add_class::<tools::UniqueIdentity>()?;
    tools_mod.add_class::<tools::UniqueKey>()?;
    tools_mod.add_class::<tools::Intersperse>()?;
    tools_mod.add_class::<tools::MapWindow>()?;
    tools_mod.add_class::<tools::MapJuxt>()?;
    tools_mod.add_class::<tools::FilterMap>()?;
    tools_mod.add_class::<tools::FilterMapStar>()?;
    tools_mod.add_class::<tools::Scan>()?;
    tools_mod.add_class::<tools::MapWhile>()?;
    tools_mod.add_class::<tools::FromFn>()?;
    tools_mod.add_class::<tools::Drain>()?;
    tools_mod.add_class::<tools::ExtractIf>()?;
    tools_mod.add_class::<tools::Successors>()?;
    tools_mod.add_class::<tools::FilterStar>()?;
    tools_mod.add_class::<tools::WithPosition>()?;
    tools_mod.add_class::<tools::ZipLongest>()?;
    tools_mod.add_class::<tools::Unzip>()?;
    tools_mod.add_class::<tools::GroupBy>()?;
    tools_mod.add_class::<tools::Iter>()?;
    tools_mod.add_class::<tools::Peekable>()?;

    let abc_mod = PyModule::new(py, "abc")?;
    let iterable_mod = PyModule::new(py, "_iterable")?;
    iterable_mod.add_class::<abc::PyoIterable>()?;
    let iterator_mod = PyModule::new(py, "_iterator")?;
    iterator_mod.add_class::<abc::PyoIterator>()?;
    let collection_mod = PyModule::new(py, "_collection")?;

    collection_mod.add_class::<abc::PyoContainer>()?;
    collection_mod.add_class::<abc::PyoSized>()?;
    collection_mod.add_class::<abc::PyoCollection>()?;
    let sequences_mod = PyModule::new(py, "_sequences")?;

    sequences_mod.add_class::<abc::PyoReversible>()?;
    sequences_mod.add_class::<abc::PyoSequence>()?;
    sequences_mod.add_class::<abc::PyoMutableSequence>()?;
    let sets_mod = PyModule::new(py, "_sets")?;
    sets_mod.add_class::<abc::PyoSet>()?;
    sets_mod.add_class::<abc::PyoMutableSet>()?;
    let mappings_mod = PyModule::new(py, "_mappings")?;

    mappings_mod.add_class::<abc::PyoMappingView>()?;
    mappings_mod.add_class::<abc::PyoMapping>()?;
    mappings_mod.add_class::<abc::PyoKeysView>()?;
    mappings_mod.add_class::<abc::PyoValuesView>()?;
    mappings_mod.add_class::<abc::PyoItemsView>()?;
    mappings_mod.add_class::<abc::PyoMutableMapping>()?;
    abc_mod.add_submodule(&iterable_mod)?;
    abc_mod.add_submodule(&iterator_mod)?;
    abc_mod.add_submodule(&collection_mod)?;
    abc_mod.add_submodule(&sequences_mod)?;
    abc_mod.add_submodule(&sets_mod)?;
    abc_mod.add_submodule(&mappings_mod)?;
    m.add_submodule(&abc_mod)?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    sys_mods.set_item("pyochain._range", range_mod)?;
    sys_mods.set_item("pyochain._tools", tools_mod)?;
    sys_mods.set_item("pyochain._seq", seq_mod)?;
    sys_mods.set_item("pyochain._vec", vec_mod)?;
    sys_mods.set_item("pyochain._set", set_mod)?;
    sys_mods.set_item("pyochain.abc._iterable", iterable_mod)?;
    sys_mods.set_item("pyochain.abc._iterator", iterator_mod)?;
    sys_mods.set_item("pyochain.abc._collection", collection_mod)?;
    sys_mods.set_item("pyochain.abc._sequences", sequences_mod)?;
    sys_mods.set_item("pyochain.abc._sets", sets_mod)?;
    sys_mods.set_item("pyochain.abc._mappings", mappings_mod)?;
    register_all(py)
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
