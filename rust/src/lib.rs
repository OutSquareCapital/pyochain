mod abc;
mod collections;
mod errors;
mod hasher;
mod mixins;
mod option;
mod pyo3_ext;
mod result;
mod seq;
mod sliceview;
mod tools;

use crate::pyo3_ext::{
    prelude::*,
    types::{PyAbstractSet, PyIterable, PyMutableSequence},
};
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
                args: &pyo3_ext::args::Args<'_>,
                kwargs: Option<&pyo3_ext::args::Kwargs<'_>>,
            ) -> PyResult<Py<PyAny>> {
                (
                    pyo3_ext::args::Concatenate::concat(func, &slf, args, kwargs)?.unbind().pipe(Ok)
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
        args: &pyo3_ext::args::Args<'_>,
        kwargs: Option<&pyo3_ext::args::Kwargs<'_>>,
    ) -> PyResult<Py<PyAny>> {
        pyo3_ext::args::Concatenate::concat(f, &slf, args, kwargs)?;
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
    m.add_class::<seq::Range>()?;
    m.add_class::<seq::Seq>()?;
    m.add_class::<seq::PyoVec>()?;
    m.add_class::<seq::Set>()?;
    m.add_class::<seq::SetMut>()?;
    m.add_class::<seq::Dict>()?;
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
    m.add_class::<tools::Peekable>()?;
    m.add_class::<sliceview::SliceView>()?;

    let abc_mod = PyModule::new(py, "abc")?;
    abc_mod.add_class::<mixins::Checkable>()?;
    abc_mod.add_class::<mixins::Fluent>()?;
    abc_mod.add_class::<mixins::PyoPipe>()?;
    abc_mod.add_class::<mixins::PyoTap>()?;
    abc_mod.add_class::<abc::PyoIterable>()?;
    abc_mod.add_class::<abc::PyoIterator>()?;
    abc_mod.add_class::<abc::PyoContainer>()?;
    abc_mod.add_class::<abc::PyoSized>()?;
    abc_mod.add_class::<abc::PyoCollection>()?;
    abc_mod.add_class::<abc::PyoReversible>()?;
    abc_mod.add_class::<abc::PyoSequence>()?;
    abc_mod.add_class::<abc::PyoMutableSequence>()?;
    abc_mod.add_class::<abc::PyoSet>()?;
    abc_mod.add_class::<abc::PyoMutableSet>()?;
    abc_mod.add_class::<abc::PyoMappingView>()?;
    abc_mod.add_class::<abc::PyoMapping>()?;
    abc_mod.add_class::<abc::PyoKeysView>()?;
    abc_mod.add_class::<abc::PyoValuesView>()?;
    abc_mod.add_class::<abc::PyoItemsView>()?;
    abc_mod.add_class::<abc::PyoMutableMapping>()?;
    m.add_submodule(&abc_mod)?;
    //TODO: Don't forget to add `collections` module once the rust migration is complete.
    m.add_class::<collections::StableSet>()?;
    m.add_class::<collections::Deque>()?;
    m.add_class::<collections::PyoCounter>()?;
    m.add_class::<collections::Heap>()?;
    m.add_class::<collections::HeapMax>()?;
    m.add_class::<collections::HeapMin>()?;
    // NOTE: Temp utils
    m.add_class::<collections::InnerLists>()?;
    m.add_class::<collections::InnerKeyLists>()?;
    m.add_function(wrap_pyfunction!(collections::bisect::bisect_left, m)?)?;
    m.add_function(wrap_pyfunction!(collections::bisect::bisect_right, m)?)?;
    m.add_function(wrap_pyfunction!(collections::bisect::insort_right, m)?)?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    sys_mods.set_item("pyochain.abc", abc_mod)?;
    register_all(py)
}
fn register_all(py: Python<'_>) -> PyResult<()> {
    let abc_mod = py.import("collections.abc")?;
    PyIterable::register::<abc::PyoIterable>(py)?;
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
    PyMutableSequence::register::<abc::PyoMutableSequence>(py)?;
    PyAbstractSet::register::<abc::PyoSet>(py)?;
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
