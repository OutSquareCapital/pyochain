use either::Either;
use pyo3::{
    PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyRange, PySet, PyTuple},
};

use crate::{collections, seq};
/// Ergonomic "flat" match of one or more `Bound<'_, PyAny>`-like values against
/// concrete pyo3 types, using [`Bound::cast`](https://docs.rs/pyo3/latest/pyo3/struct.Bound.html#method.cast).
///
/// The values are listed once in the header, then each arm lists the target type
/// for every value. On success, the arm's block runs with each identifier shadowed
/// as `&Bound<'_, Ty>`. If any cast in an arm fails, matching falls through to the
/// next arm, exactly like a regular `match`. The final arm must be `_ => { .. }`.
///
/// # Example
/// ```ignore
/// cast_match!((inner, index) {
///     (PyList, PySlice) => { /* inner: &Bound<'_, PyList>, index: &Bound<'_, PySlice> */ },
///     (PyList, PyInt) => { /* inner: &Bound<'_, PyList>, index: &Bound<'_, PyInt> */ },
///     _ => { /* fallback, original bindings untouched */ },
/// })
/// ```
#[macro_export]
macro_rules! cast_match {
    (($($val:ident),+ $(,)?) { $($arms:tt)* }) => {
        $crate::cast_match!(@arms ($($val),+) { $($arms)* })
    };
    ($val:ident { $($arms:tt)* }) => {
        $crate::cast_match!(@arms ($val) { $($arms)* })
    };
    (@arms ($($val:ident),+) {
        ($($ty:ty),+) => $body:block,
        $($rest:tt)*
    }) => {
        $crate::cast_match!(@build
            ($($val),+);
            ($($val),+);
            ($($ty),+);
            (); ();
            $body;
            { $($rest)* }
        )
    };
    (@arms ($($val:ident),+) {
        $ty:ty => $body:block,
        $($rest:tt)*
    }) => {
        $crate::cast_match!(@build
            ($($val),+);
            ($($val),+);
            ($ty);
            (); ();
            $body;
            { $($rest)* }
        )
    };
    (@arms ($($val:ident),+) {
        _ => $default:block $(,)?
    }) => {
        $default
    };
    (@build
        ($original_val:ident $(, $original_rest:ident)*);
        ($val:ident $(, $rest_val:ident)*);
        ($ty:ty $(, $rest_ty:ty)*);
        ($($expr:expr),*);
        ($($pat:pat),*);
        $body:block;
        { $($rest:tt)* }
    ) => {
        $crate::cast_match!(@build
            ($original_val $(, $original_rest)*);
            ($($rest_val),*);
            ($($rest_ty),*);
            ($($expr,)* $val.cast::<$ty>());
            ($($pat,)* Ok($val));
            $body;
            { $($rest)* }
        )
    };
    (@build
        ($($original_val:ident),+);
        ();
        ();
        ($($expr:expr),+);
        ($($pat:pat),+);
        $body:block;
        { $($rest:tt)* }
    ) => {
        match ($($expr),*) {
            ($($pat),*) => $body,
            _ => $crate::cast_match!(@arms
                ($($original_val),+) { $($rest)* }
            ),
        }
    };
}
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
