use pyo3::{
    PyTypeCheck, ffi,
    prelude::*,
    types::{PyDict, PyType},
};

use derive_more::{Constructor, From};
use pyo3::sync::PyOnceLock;
type Builtin = PyBuiltin<Py<PyAny>>;
type Class = PyBuiltin<Py<PyType>>;

#[derive(Debug, Clone, Copy, Constructor, From)]
struct PyStdMod(&'static str);
impl PyStdMod {
    pub const fn register<T>(self, name: &'static str) -> PyBuiltin<T> {
        PyBuiltin::new(self.0, name)
    }
}
pub(crate) struct PyBuiltin<T> {
    inner: PyOnceLock<T>,
    module: &'static str,
    name: &'static str,
}
impl<T> PyBuiltin<T> {
    #[inline(always)]
    const fn new(module: &'static str, name: &'static str) -> Self {
        Self {
            inner: PyOnceLock::new(),
            module,
            name,
        }
    }
}
impl<T: PyTypeCheck> PyBuiltin<Py<T>> {
    #[inline(always)]
    pub fn import<'py>(&self, py: Python<'py>) -> &Bound<'py, T> {
        self.inner
            .import(py, self.module, self.name)
            .expect("Failed to import {module}::{name}")
    }
}
impl PyBuiltin<Py<PyType>> {
    #[inline(always)]
    pub fn type_object_raw(&self, py: Python<'_>) -> *mut ffi::PyTypeObject {
        self.import(py).as_type_ptr()
    }
}
mod modules {
    use super::*;
    pub const BUILTINS: PyStdMod = PyStdMod::new("builtins");
    pub const FUNCTOOLS: PyStdMod = PyStdMod::new("functools");
    pub const ITERTOOLS: PyStdMod = PyStdMod::new("itertools");
    pub const SYS: PyStdMod = PyStdMod::new("sys");
    pub const HEAPQ: PyStdMod = PyStdMod::new("heapq");
    pub const OPERATOR: PyStdMod = PyStdMod::new("operator");
    pub const COLLECTIONS: PyStdMod = PyStdMod::new("collections");
    pub const COLLECTIONS_ABC: PyStdMod = PyStdMod::new("collections.abc");
    pub const TYPING: PyStdMod = PyStdMod::new("typing");
}

pub(crate) mod typing {
    use super::*;
    pub static SUPPORTS_INDEX: Class = modules::TYPING.register("SupportsIndex");
}
pub(super) mod builtins {
    use super::*;
    pub static OBJECT: Builtin = modules::BUILTINS.register("object");
    pub static ALL: Builtin = modules::BUILTINS.register("all");
    pub static ANY: Builtin = modules::BUILTINS.register("any");
    pub static MAX: Builtin = modules::BUILTINS.register("max");
    pub static MIN: Builtin = modules::BUILTINS.register("min");
    pub static SUM: Builtin = modules::BUILTINS.register("sum");
    pub static ENUMERATE: Builtin = modules::BUILTINS.register("enumerate");
    pub static MAP: Builtin = modules::BUILTINS.register("map");
    pub static FILTER: Builtin = modules::BUILTINS.register("filter");
    pub static SORTED: Builtin = modules::BUILTINS.register("sorted");
    pub static ZIP: Builtin = modules::BUILTINS.register("zip");
}
pub(crate) mod itertools {
    use super::*;
    pub static ACCUMULATE: Builtin = modules::ITERTOOLS.register("accumulate");
    pub static BATCHED: Builtin = modules::ITERTOOLS.register("batched");
    pub static TEE: Builtin = modules::ITERTOOLS.register("tee");
    pub static GROUP_BY: Builtin = modules::ITERTOOLS.register("groupby");
    pub static ZIP_LONGEST: Builtin = modules::ITERTOOLS.register("zip_longest");
    pub static FILTER_FALSE: Builtin = modules::ITERTOOLS.register("filterfalse");
    pub static MAP_STAR: Builtin = modules::ITERTOOLS.register("starmap");
    pub static COUNT: Builtin = modules::ITERTOOLS.register("count");
    pub static TAKE_WHILE: Builtin = modules::ITERTOOLS.register("takewhile");
    pub static DROP_WHILE: Builtin = modules::ITERTOOLS.register("dropwhile");
    pub static COMBINATIONS: Builtin = modules::ITERTOOLS.register("combinations");
    pub static COMBINATIONS_WITH_REPLACEMENT: Builtin =
        modules::ITERTOOLS.register("combinations_with_replacement");
    pub static COMPRESS: Builtin = modules::ITERTOOLS.register("compress");
    pub static CYCLE: Builtin = modules::ITERTOOLS.register("cycle");
    pub static PAIRWISE: Builtin = modules::ITERTOOLS.register("pairwise");
    pub static PRODUCT: Builtin = modules::ITERTOOLS.register("product");
    pub static PERMUTATIONS: Builtin = modules::ITERTOOLS.register("permutations");
    pub static ISLICE: Builtin = modules::ITERTOOLS.register("islice");
    pub static CHAIN: Builtin = modules::ITERTOOLS.register("chain");
    pub static REPEAT: Class = modules::ITERTOOLS.register("repeat");
}
pub(super) mod heapq {
    use super::*;
    pub static MERGE: Builtin = modules::HEAPQ.register("merge");
    pub static NLARGEST: Builtin = modules::HEAPQ.register("nlargest");
    pub static NSMALLEST: Builtin = modules::HEAPQ.register("nsmallest");
    pub static HEAPIFY_MAX: Builtin = modules::HEAPQ.register("_heapify_max");
    pub static HEAPIFY: Builtin = modules::HEAPQ.register("heapify");
    pub static HEAPPOP: Builtin = modules::HEAPQ.register("heappop");
    pub static HEAPPUSH: Builtin = modules::HEAPQ.register("heappush");
    pub static HEAPPUSHPOP: Builtin = modules::HEAPQ.register("heappushpop");
    pub static HEAPREPLACE: Builtin = modules::HEAPQ.register("heapreplace");
}
pub mod collections {
    use super::*;
    pub static DEQUE: Class = modules::COLLECTIONS.register("deque");
    pub mod abc {
        use super::*;
        pub static SIZED: Class = modules::COLLECTIONS_ABC.register("Sized");
        pub static ITERABLE: Class = modules::COLLECTIONS_ABC.register("Iterable");
        pub static MAPPING_VIEW: Class = modules::COLLECTIONS_ABC.register("MappingView");
        pub static SET: Class = modules::COLLECTIONS_ABC.register("Set");
        pub static MUTABLE_SEQUENCE: Class = modules::COLLECTIONS_ABC.register("MutableSequence");
        pub static MUTABLE_MAPPING: Class = modules::COLLECTIONS_ABC.register("MutableMapping");
        pub static MUTABLE_SET: Class = modules::COLLECTIONS_ABC.register("MutableSet");
        pub static KEYS_VIEW: Class = modules::COLLECTIONS_ABC.register("KeysView");
        pub static VALUES_VIEW: Class = modules::COLLECTIONS_ABC.register("ValuesView");
        pub static ITEMS_VIEW: Class = modules::COLLECTIONS_ABC.register("ItemsView");
        pub static CONTAINER: Class = modules::COLLECTIONS_ABC.register("Container");
        pub static COLLECTION: Class = modules::COLLECTIONS_ABC.register("Collection");
        pub static REVERSIBLE: Class = modules::COLLECTIONS_ABC.register("reversible");
        pub static SUPPORTS_ITEMS: Class = modules::COLLECTIONS_ABC.register("SupportsItems");
    }
}

pub(super) static REDUCE: Builtin = modules::FUNCTOOLS.register("reduce");
pub(super) static MODULES: PyBuiltin<Py<PyDict>> = modules::SYS.register("modules");
pub(super) static ITEMGETTER: Builtin = modules::OPERATOR.register("itemgetter");
