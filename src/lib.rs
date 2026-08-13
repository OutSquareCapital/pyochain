mod abc;
mod collections;
mod core;
mod display;
mod errors;
mod hasher;
mod traits;
use crate::collections::sorted::debug;
use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{PyIterator, PyMapping, PySequence, PyType},
};
use pyo3_ext::{
    prelude::*,
    types::{PyAbstractSet, PyIterable, PyMappingView, PyMutableSequence, PyMutableSet},
};
#[pymodule]
fn pyochain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    debug_backtrace();
    let py = m.py();
    let abc_mod = PyModule::new(py, "abc")?;
    let collections_mod = PyModule::new(py, "collections")?;
    let sorted_mod = PyModule::new(py, "_sorted")?;
    let sys_mods = py.import("sys")?.getattr("modules")?;
    m.add_submodule(&abc_mod)?;
    m.add_submodule(&collections_mod)?;
    collections_mod.add_submodule(&sorted_mod)?;
    populate_core(m, py)?;
    populate_abc(&abc_mod)?;
    populate_collections(&collections_mod)?;
    populate_sorted(&sorted_mod)?;
    sys_mods.set_item("pyochain", m)?;
    sys_mods.set_item("pyochain.abc", abc_mod)?;
    sys_mods.set_item("pyochain.collections", collections_mod)?;
    sys_mods.set_item("pyochain.collections._sorted", sorted_mod)?;
    register_all(py)
}
fn populate_core(m: &Bound<'_, PyModule>, py: Python<'_>) -> PyResult<()> {
    core::PyNull::init(py)?;
    m.add_class::<core::PyochainOption>()?;
    m.add_class::<core::PyochainOptionType>()?;
    m.add_class::<core::PySome>()?;
    m.add_class::<core::PyNull>()?;
    m.add_function(wrap_pyfunction!(core::then_if_some, m)?)?;
    m.add_function(wrap_pyfunction!(core::then_if_true, m)?)?;
    m.add_function(wrap_pyfunction!(core::new_option, m)?)?;
    m.add("NONE", core::PyNull::get(py))?;
    m.add_class::<core::PyoOk>()?;
    m.add_class::<core::PyoErr>()?;
    m.add_class::<errors::OptionUnwrapError>()?;
    m.add_class::<errors::ResultUnwrapError>()?;
    m.add_class::<core::PyochainResult>()?;
    m.add_class::<core::PyochainResultType>()?;
    m.add_class::<core::Range>()?;
    m.add_class::<core::Seq>()?;
    m.add_class::<core::PyoVec>()?;
    m.add_class::<core::Set>()?;
    m.add_class::<core::SetMut>()?;
    m.add_class::<core::Dict>()?;
    m.add_class::<core::iterators::Iter>()?;
    m.add_class::<core::iterators::Peekable>()?;
    m.add_class::<core::SliceView>()
}
fn populate_abc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<abc::Checkable>()?;
    m.add_class::<abc::Fluent>()?;
    m.add_class::<abc::PyoPipe>()?;
    m.add_class::<abc::PyoTap>()?;
    m.add_class::<abc::PyoIterable>()?;
    m.add_class::<abc::PyoIterator>()?;
    m.add_class::<abc::PyoContainer>()?;
    m.add_class::<abc::PyoSized>()?;
    m.add_class::<abc::PyoCollection>()?;
    m.add_class::<abc::PyoReversible>()?;
    m.add_class::<abc::PyoSequence>()?;
    m.add_class::<abc::PyoMutableSequence>()?;
    m.add_class::<abc::PyoSet>()?;
    m.add_class::<abc::PyoMutableSet>()?;
    m.add_class::<abc::PyoMappingView>()?;
    m.add_class::<abc::PyoMapping>()?;
    m.add_class::<abc::PyoKeysView>()?;
    m.add_class::<abc::PyoValuesView>()?;
    m.add_class::<abc::PyoItemsView>()?;
    m.add_class::<abc::PyoMutableMapping>()
}
fn populate_collections(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<collections::StableSet>()?;
    m.add_class::<collections::Deque>()?;
    m.add_class::<collections::PyoCounter>()?;
    m.add_class::<collections::Heap>()?;
    m.add_class::<collections::HeapMax>()?;
    m.add_class::<collections::HeapMin>()?;
    m.add_class::<collections::SortedList>()?;
    m.add_class::<collections::SortedKeyList>()?;
    m.add_class::<collections::SortedSet>()?;
    m.add_class::<collections::SortedKeySet>()?;
    m.add_class::<collections::SortedDict>()?;
    m.add_class::<collections::SortedKeyDict>()?;
    m.add_class::<collections::sorted::SortedKeysView>()?;
    m.add_class::<collections::sorted::SortedValuesView>()?;
    m.add_class::<collections::sorted::SortedItemsView>()
}
fn populate_sorted(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(debug::check_sorted_list, m)?)?;
    m.add_function(wrap_pyfunction!(debug::check_sorted_key_list, m)?)?;
    m.add_function(wrap_pyfunction!(debug::assert_sorted_list_empty, m)?)?;
    m.add_function(wrap_pyfunction!(debug::check_sorted_set, m)?)?;
    m.add_function(wrap_pyfunction!(debug::check_sorted_dict, m)?)
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
    PyMappingView::register::<abc::PyoMappingView>(py)?;
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
