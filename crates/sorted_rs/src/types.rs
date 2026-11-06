use parking_lot::RwLock;

use either::Either;
use pyo3::{
    prelude::*,
    types::{PyDict, PyInt, PyList, PySequence, PySet, PySlice},
};

use crate::Collection;

/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;

// Either types for various Python objects.

pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;
pub type ListOrAny<'py> = Either<Bound<'py, PyList>, Bound<'py, PyAny>>;
pub type IntOrSlice<'py> = Either<Bound<'py, PyInt>, Bound<'py, PySlice>>;
pub type Dict<T> = Collection<T, PyDict>;
pub type Set<T> = Collection<T, PySet>;
pub type DictRef<T> = Py<RwLock<Dict<T>>>;
