use either::Either;
use pyo3::{
    prelude::*,
    types::{PyList, PySequence, PySlice},
};

/// A `Vec` which contains python objects.
pub type VecPy = Vec<Py<PyAny>>;

// Either types for various Python objects.

pub type SeqOrAny<'py> = Either<Bound<'py, PySequence>, Bound<'py, PyAny>>;
pub type ListOrAny<'py> = Either<Bound<'py, PyList>, Bound<'py, PyAny>>;
pub type IntOrSlice<'py> = Either<isize, Bound<'py, PySlice>>;
