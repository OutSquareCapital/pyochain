mod abc;
mod collections;
mod dict;
mod display;
mod errors;
mod hasher;
mod iterators;
mod option;
mod pyo3_ext;
mod pyovec;
mod range;
mod result;
mod seq;
mod sets;
mod sliceview;
mod traits;
use crate::pyo3_ext::{
    prelude::*,
    types::{PyAbstractSet, PyIterable, PyMutableSequence, PyMutableSet},
};
use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{PyIterator, PyMapping, PySequence, PyType},
};

#[pymodule]
fn rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    debug_backtrace();
    let py = m.py();
    option::PyNull::init(py)?;
    m.add_class::<option::PyochainOption>()?;
    m.add_class::<option::PyochainOptionType>()?;
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
    m.add_class::<result::PyochainResultType>()?;
    m.add_class::<range::Range>()?;
    m.add_class::<seq::Seq>()?;
    m.add_class::<pyovec::PyoVec>()?;
    m.add_class::<sets::Set>()?;
    m.add_class::<sets::SetMut>()?;
    m.add_class::<dict::Dict>()?;
    m.add_class::<iterators::Iter>()?;
    m.add_class::<iterators::Peekable>()?;
    m.add_class::<sliceview::SliceView>()?;

    let abc_mod = PyModule::new(py, "abc")?;
    abc_mod.add_class::<abc::Checkable>()?;
    abc_mod.add_class::<abc::Fluent>()?;
    abc_mod.add_class::<abc::PyoPipe>()?;
    abc_mod.add_class::<abc::PyoTap>()?;
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
    m.add_function(wrap_pyfunction!(
        collections::sorted::debug::check_sorted_list,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        collections::sorted::debug::check_sorted_key_list,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        collections::sorted::debug::assert_sorted_list_empty,
        m
    )?)?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    sys_mods.set_item("pyochain.abc", abc_mod)?;
    register_all(py)
}
fn register_all(py: Python<'_>) -> PyResult<()> {
    let abc_mod = py.import("collections.abc")?;
    PyIterable::register::<abc::PyoIterable>(py)?;
    PyIterator::register::<abc::PyoIterator>(py)?;
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
    PyMutableSet::register::<abc::PyoMutableSet>(py)?;
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
#[cfg(debug_assertions)]
fn debug_backtrace() {
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "full");
    }
    color_eyre::install().unwrap();
}

#[cfg(not(debug_assertions))]
fn debug_backtrace() {}
