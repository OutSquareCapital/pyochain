use std::sync::{Arc, Mutex};

use either::Either;
use pyo3::{
    prelude::*,
    types::{PyInt, PyList, PySequence, PySlice},
};

use crate::DictData;

/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;

// Either types for various Python objects.

pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;
pub type ListOrAny<'py> = Either<Bound<'py, PyList>, Bound<'py, PyAny>>;
pub type IntOrSlice<'py> = Either<Bound<'py, PyInt>, Bound<'py, PySlice>>;

pub type DictDataRef<T> = Arc<Mutex<DictData<T>>>;
