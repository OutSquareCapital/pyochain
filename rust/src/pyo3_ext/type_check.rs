use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyRange, PySet, PyTuple},
};

use crate::{collections, seq};
/// Ergonomic "flat" match of one or more `Bound<'_, PyAny>`-like values against concrete pyo3 types, using [`Bound::cast`].\
/// This avoids awkward nested `match` statements, or verbose `if let` chains.\
/// ## Current limitations
/// - Unfortunately, we still have to use `match` keyword systematically, otherwise rustfmt will completely skip the code inside the macro.
/// - For some reason, rustfmt will delete `,` statements between the match arms, so we need to handle that specially without requiring them, which diverge from regular rust syntax.
///
/// ## Example
/// ```rust
/// use pyo3::prelude::*;
/// use pyo3::types::{PyDict, PyList, PyTuple};
/// use pyo3_ext::try_cast;
/// use pyo3::exceptions::PyTypeError;
///
/// fn foo(value: Bound<'_, PyAny>) -> PyResult<isize> {
///   try_cast!(match value {
///     PyList | PyTuple => { Ok(0) }
///     PyDict => { Ok(1) }
///     _ => { Err(PyTypeError::new_err("Invalid type")) }
///    })
///  }
/// fn foo_no_macro(value: Bound<'_, PyAny>) -> PyResult<()> {
///   match value.cast::<PyList>() {
///     Ok(_) => Ok(0),
///     Err(_) => match value.cast::<PyTuple>() {
///         Ok(_) => Ok(1),
///         Err(_) => match value.cast::<PyDict>() {
///             Ok(_) => Ok(2),
///             Err(_) => Err(PyTypeError::new_err("Invalid type")),
///         }
///     }
/// }
/// ```
#[macro_export]
macro_rules! try_cast {
    (match ($($value:ident),+ $(,)?) { $($cases:tt)* }) => {
        $crate::try_cast!(@cases ($($value),+) { $($cases)* })
    };
    (match $value:ident { $($cases:tt)* }) => {
        $crate::try_cast!(@cases ($value) { $($cases)* })
    };
    (@cases ($($value:ident),+) {
        _ => $body:block $(,)?
    }) => {
        $body
    };
    (@cases ($($value:ident),+) {
        ($($ty:ty),+) => $body:block $(,)? $($rest:tt)*
    }) => {
        $crate::try_cast!(@values
            ($($value),+)
            ($($ty),+)
            $body
            {
                $crate::try_cast!(@cases ($($value),+) { $($rest)* })
            }
        )
    };
    (@cases ($value:ident) {
        $($ty:ty)|+ => $body:block $(,)? $($rest:tt)*
    }) => {
        $crate::try_cast!(@types
            $value
            ($($ty)|+)
            $body
            {
                $crate::try_cast!(@cases ($value) { $($rest)* })
            }
        )
    };
    (@values ($value:ident) ($ty:ty) $body:block $fallback:block) => {
        $crate::try_cast!(@types $value ($ty) $body $fallback)
    };
    (@values
        ($value:ident, $($rest_value:ident),+)
        ($ty:ty, $($rest_ty:ty),+)
        $body:block
        $fallback:block
    ) => {
        match $value.cast::<$ty>() {
            Ok($value) => $crate::try_cast!(@values
                ($($rest_value),+)
                ($($rest_ty),+)
                $body
                $fallback
            ),
            Err(_) => $fallback,
        }
    };
    (@types $value:ident ($ty:ty) $body:block $fallback:block) => {
        match $value.cast::<$ty>() {
            Ok($value) => $body,
            Err(_) => $fallback,
        }
    };
    (@types $value:ident ($ty:ty | $($rest_ty:ty)|+) $body:block $fallback:block) => {
        match $value.cast::<$ty>() {
            Ok($value) => $body,
            Err(_) => $crate::try_cast!(@types
                $value
                ($($rest_ty)|+)
                $body
                $fallback
            ),
        }
    };
}
pub trait PyWrapper: PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync {
    type Inner: PyTypeInfo;
    fn as_inner(&self) -> &Py<Self::Inner>;
    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<&'r Bound<'py, Self::Inner>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().as_inner().bind(py))
            .or_else(|_| value.cast_exact::<Self::Inner>())
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
/// Implement `PyWrapper` for pyochain types in one line.
macro_rules! impl_py_wrapper {
    ($($wrapper:ty => $inner:ty),* $(,)?) => {
        $(
            impl PyWrapper for $wrapper {
                type Inner = $inner;

                #[inline]
                fn as_inner(&self) -> &Py<Self::Inner> {
                    &self.inner
                }
            }
        )*
    };
}
impl_py_wrapper! {
    seq::Seq => PyTuple,
    seq::Vec => PyList,
    seq::Set => PyFrozenSet,
    seq::SetMut => PySet,
    seq::Range => PyRange,
    seq::Dict => PyDict,
    collections::StableSet => PyDict,
    collections::PyoCounter => PyDict,
}
