use crate::{collections, seq};
use either::Either;
use pyo3::exceptions::PyTypeError;
use pyo3::{
    PyTypeInfo,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyRange, PySet, PyTuple},
};

pub trait PyWrapper: PyTypeInfo {
    type Inner: PyTypeInfo;

    /// Extract the type of a value and check if it is one of two types, returning an `Either` with the result.\
    /// Returns a `PyErr` if the value is not one of the two types.
    /// For example, if `T` is `Vec`, this function will check if the value is a `Vec` or a `PyList`, and return either a `Ok(Vec)`, `Ok(PyList)`, or a `Err(PyTypeError)`.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<Either<&'r Bound<'py, Self>, &'r Bound<'py, Self::Inner>>> {
        value
            .cast_exact::<Self>()
            .map(Either::Left)
            .or_else(|_| value.cast_exact::<Self::Inner>().map(Either::Right))
            .map_err(|_| {
                let py = value.py();
                let wrapper_name = Self::type_object(py).name().unwrap();
                let inner_name = Self::Inner::type_object(py).name().unwrap();
                let value_name = value.get_type().name().unwrap();
                let txt = format!(
                    "Input must be a '{}'' or a '{}', got '{}'",
                    wrapper_name, inner_name, value_name
                );
                PyTypeError::new_err(txt)
            })
    }
}

impl PyWrapper for seq::Seq {
    type Inner = PyTuple;
}
impl PyWrapper for seq::Vec {
    type Inner = PyList;
}
impl PyWrapper for seq::Set {
    type Inner = PyFrozenSet;
}
impl PyWrapper for seq::SetMut {
    type Inner = PySet;
}
impl PyWrapper for seq::Range {
    type Inner = PyRange;
}
impl PyWrapper for seq::Dict {
    type Inner = PyDict;
}
impl PyWrapper for collections::StableSet {
    type Inner = PyDict;
}
