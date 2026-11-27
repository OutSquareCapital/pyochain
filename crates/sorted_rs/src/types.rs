use parking_lot::RwLock;

use either::Either;
use pyo3::{
    prelude::*,
    types::{PyDict, PyList, PySequence, PySet, PySlice},
};

use crate::Collection;

/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;
/// A pair of items of the same type.
pub type Pair<T> = (T, T);
// Either types for various Python objects.
type AnyOr<'py, T> = Either<Bound<'py, T>, Bound<'py, PyAny>>;
pub type SeqOrAny<'py> = AnyOr<'py, PySequence>;
/// Arguments for an indexing (list[...]) operations
pub type IntoIndex<'py> = Either<Bound<'py, PySlice>, isize>;
/// Return type from an indexing operation
pub type IndexOut<'py> = AnyOr<'py, PyList>;
pub type Dict<T> = Collection<T, PyDict>;
pub type Set<T> = Collection<T, PySet>;
/// A sorted dict coming from python
pub type DictRef<T> = Py<RwLock<Dict<T>>>;
