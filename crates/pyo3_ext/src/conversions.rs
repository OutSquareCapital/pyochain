use pyo3::{PyTypeInfo, prelude::*, types};

use crate::types::PyDeque;
/// Trait for types that we know can safely be converted into a `PyIterator` (i.e. they implement the `__iter__` method in Python).
pub trait IntoPyIterator<'py> {
    fn iter_py(&self) -> Bound<'py, types::PyIterator>;
}
impl<'py> IntoPyIterator<'py> for Bound<'py, types::PyIterator> {
    fn iter_py(&self) -> Bound<'py, types::PyIterator> {
        self.to_owned()
    }
}
macro_rules! impl_into_py_iterator_for_iterable {
    ($($t:ty),* $(,)?) => {
        $(
            impl<'py> IntoPyIterator<'py> for Bound<'py, $t> {
                /// Returns a `PyIterator` with `unwrap_unchecked`, as we know that the type implements `__iter__` and thus can be safely converted into a `PyIterator`.
                fn iter_py(&self) -> Bound<'py, types::PyIterator> {
                    unsafe { self.try_iter().unwrap_unchecked() }
                }
            }
        )*
    };
}
impl_into_py_iterator_for_iterable!(
    types::PyTuple,
    types::PyList,
    types::PySet,
    types::PyDict,
    types::PyFrozenSet,
    types::PyRange,
    PyDeque,
    types::PyDictKeys,
    types::PyDictValues,
    types::PyDictItems,
    types::PySequence,
);

/// Trait for types that can be converted from another type via their class constructors.\
/// The default implementation of `try_from_py` calls the type's constructor with the object as an argument, and casts the result into the target type.\
pub trait TryFromPy<T: PyTypeInfo = types::PyAny>: Sized + PyTypeInfo {
    #[inline(always)]
    fn try_from_py(obj: Bound<'_, T>) -> PyResult<Bound<'_, Self>> {
        Self::type_object(obj.py())
            .call1((obj,))
            .map(|t| unsafe { t.cast_into_unchecked::<Self>() })
    }
}
/// Counterpart of `TryFromPy`, for types that can be converted into another type via a Python call.\
pub trait TryIntoPy<'py, T: PyTypeInfo> {
    fn try_into_py<I: TryFromPy<T>>(self) -> PyResult<Bound<'py, I>>;
}
impl<'py, T: Sized + PyTypeInfo> TryIntoPy<'py, T> for Bound<'py, T> {
    #[inline(always)]
    fn try_into_py<I: TryFromPy<T>>(self) -> PyResult<Bound<'py, I>> {
        I::try_from_py(self)
    }
}
macro_rules! impl_default_try_from_py {
    ($($target:ty),* $(,)?) => {
        $(
            impl<T: PyTypeInfo> TryFromPy<T> for $target {}
        )*
    };
}

impl_default_try_from_py!(
    types::PyTuple,
    types::PyList,
    types::PySet,
    types::PyFrozenSet,
    PyDeque,
    types::PyDict,
    types::PyString,
);
