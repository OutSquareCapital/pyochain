mod abc;
mod collections;
mod constructors;
mod core;
mod mixins;
mod traits;
use crate::collections::sorted::debug;
use pyo3::{
    prelude::*,
    types::{PyIterator, PyMapping, PySequence},
};
use pyo3_ext::{
    prelude::*,
    pylibs,
    types::{
        PyAbstractSet, PyCollection, PyContainer, PyItemsView, PyIterable, PyKeysView,
        PyMappingView, PyMutableMapping, PyMutableSequence, PyMutableSet, PyReversible, PySized,
        PyValuesView,
    },
};
use sorted_rs::iter::PySortedIter;

const PACKAGE: &str = "pyochain";

#[pymodule]
fn pyochain(m: &Bound<'_, PyModule>) -> PyResult<()> {
    debug_backtrace();
    pylibs::sys::add_module(PACKAGE, m)?;
    Mod::init_all(m)?;
    register_all(m.py())
}
/// All the submodules of `pyochain` that need to be initialized and added to the package.
#[derive(Debug, Clone, Copy)]
enum Mod {
    Core,
    Abc,
    Collections,
    Constructors,
    Mixins,
    Sorted,
}
impl Mod {
    fn init_all(m: &Bound<'_, PyModule>) -> PyResult<()> {
        // NOTE: We need to init `core` twice to handle both relative imports, e.g `from pyochain import Vec` and direct import paths, e.g `import pyochain.core.Vec`
        Self::Core.init(m)?;
        Self::Core.add_to(m)?;
        Self::Abc.add_to(m)?;
        Self::Mixins.add_to(m)?;
        Self::Constructors.add_to(m)?;
        Self::Collections
            .add_to(m)
            .and_then(|collections| Mod::Sorted.add_to(&collections))?;
        Ok(())
    }
    const fn in_submodule(self) -> bool {
        !matches!(self, Self::Sorted)
    }
    const fn name(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Abc => "abc",
            Self::Collections => "collections",
            Self::Constructors => "constructors",
            Self::Mixins => "mixins",
            Self::Sorted => "_sorted",
        }
    }
    fn init(self, m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        match self {
            Self::Core => {
                m.add_class::<core::PyochainOption>()?;
                m.add_class::<core::PyochainOptionType>()?;
                m.add_class::<core::PySome>()?;
                m.add_class::<core::PyNull>()?;
                m.add_function(wrap_pyfunction!(core::then_if_some, m)?)?;
                m.add_function(wrap_pyfunction!(core::then_if_true, m)?)?;
                m.add_function(wrap_pyfunction!(core::new_option, m)?)?;
                core::NONE
                    .get_or_try_init(py, || Py::new(py, core::PyNull))
                    .and_then(|null| m.add("NONE", null))?;
                m.add_class::<core::PyoOk>()?;
                m.add_class::<core::PyoErr>()?;
                m.add_class::<core::OptionUnwrapError>()?;
                m.add_class::<core::ResultUnwrapError>()?;
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
            Self::Abc => {
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
            Self::Mixins => {
                m.add_class::<mixins::Checkable>()?;
                m.add_class::<mixins::Fluent>()?;
                m.add_class::<mixins::PyoPipe>()?;
                m.add_class::<mixins::PyoTap>()
            }
            Self::Constructors => {
                m.add_class::<constructors::KwargsWrapper>()?;
                m.add_class::<constructors::ArgsWrapper>()?;
                m.add_class::<constructors::Wrapper>()?;
                m.add_class::<constructors::FromKwargs>()?;
                m.add_class::<constructors::FromArgs>()?;
                m.add_class::<constructors::FromIter>()
            }
            Self::Collections => {
                collections::sorted::iter::SortedIter::install(py);
                collections::sorted::iter::SortedIterRev::install(py);
                collections::sorted::iter::SortedIterBounded::install(py);
                collections::sorted::iter::SortedIterBoundedRev::install(py);
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
            Self::Sorted => {
                m.add_function(wrap_pyfunction!(debug::check_sorted_list, m)?)?;
                m.add_function(wrap_pyfunction!(debug::check_sorted_key_list, m)?)?;
                m.add_function(wrap_pyfunction!(debug::assert_sorted_list_empty, m)?)?;
                m.add_function(wrap_pyfunction!(debug::check_sorted_set, m)?)?;
                m.add_function(wrap_pyfunction!(debug::check_sorted_dict, m)?)
            }
        }
    }
    fn add_to<'py>(self, parent: &Bound<'py, PyModule>) -> PyResult<Bound<'py, PyModule>> {
        let name = self.name();
        let full_name = if self.in_submodule() {
            format!("{PACKAGE}.{name}")
        } else {
            format!("{PACKAGE}.{}.{name}", parent.name()?)
        };
        let module = PyModule::new(parent.py(), name)?;
        parent.add_submodule(&module)?;
        pylibs::sys::add_module(&full_name, &module)?;
        self.init(&module)?;
        Ok(module)
    }
}
/// Register all the pertinent pyochain classes with the corresponding Python ABCs.
fn register_all(py: Python<'_>) -> PyResult<()> {
    PyIterable::register::<abc::PyoIterable>(py)?;
    PyIterator::register::<abc::PyoIterator>(py)?;
    PyContainer::register::<abc::PyoContainer>(py)?;
    PySized::register::<abc::PyoSized>(py)?;
    PyContainer::register::<abc::PyoCollection>(py)?;
    PySized::register::<abc::PyoCollection>(py)?;
    PyCollection::register::<abc::PyoCollection>(py)?;
    PyReversible::register::<abc::PyoReversible>(py)?;
    PyReversible::register::<abc::PyoSequence>(py)?;
    PyMappingView::register::<abc::PyoMappingView>(py)?;
    PyMutableSequence::register::<abc::PyoMutableSequence>(py)?;
    PyAbstractSet::register::<abc::PyoSet>(py)?;
    PyMutableSet::register::<abc::PyoMutableSet>(py)?;
    PySequence::register::<abc::PyoSequence>(py)?;
    PyKeysView::register::<abc::PyoKeysView>(py)?;
    PyValuesView::register::<abc::PyoValuesView>(py)?;
    PyItemsView::register::<abc::PyoItemsView>(py)?;
    PyMapping::register::<abc::PyoMapping>(py)?;
    PyMutableMapping::register::<abc::PyoMutableMapping>(py)?;
    PyKeysView::register::<collections::sorted::SortedKeysView>(py)?;
    PyValuesView::register::<collections::sorted::SortedValuesView>(py)?;
    PyItemsView::register::<collections::sorted::SortedItemsView>(py)?;
    PyKeysView::register::<collections::sorted::SortedByKeyKeysView>(py)?;
    PyValuesView::register::<collections::sorted::SortedByKeyValuesView>(py)?;
    PyItemsView::register::<collections::sorted::SortedByKeyItemsView>(py)
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
