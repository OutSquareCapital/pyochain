use pyo3::{PyClass, PyTypeInfo, prelude::*, types};

use crate::types::PyDeque;
/// Trait for types that we know can safely be converted into a `PyIterator` (i.e. they implement the `__iter__` method in Python).
pub trait IntoPyIterator<'py> {
    /// Unchecked conversion to a `PyIterator` with `unwrap_unchecked`, as we know that the type implements `__iter__` and thus can be safely converted into a `PyIterator`.
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

/// Trait for types that can be converted from another type via their class constructors.
pub trait TryFromPy<T: PyTypeInfo = types::PyAny>: PyTypeInfo {
    /// The default implementation calls the type's constructor with the object as an argument, and casts the result into the target type.\
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
impl<'py, T: PyTypeInfo> TryIntoPy<'py, T> for Bound<'py, T> {
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

pub trait IntoSequence<'py> {
    fn as_sequence(&self) -> &Bound<'py, types::PySequence>;
    fn into_sequence(self) -> Bound<'py, types::PySequence>;
}
macro_rules! impl_into_sequence {
    ($($t:ty),*) => {
        $(
            impl<'py> IntoSequence<'py> for Bound<'py, $t> {
                fn as_sequence(&self) -> &Bound<'py, types::PySequence> {
                    unsafe { self.cast_unchecked::<types::PySequence>() }
                }
                fn into_sequence(self) -> Bound<'py, types::PySequence> {
                    unsafe { self.cast_into_unchecked::<types::PySequence>() }
                }
            }
        )*
    };
}
// NOTE: We don't implement it for `PyTuple` and `PyList` because equivalent methods already exist for them.
impl_into_sequence!(PyDeque, types::PyString, types::PyRange);
/// Convenience trait to build nested hierarchies of pyclasses concisely at call sites
pub trait Abc:
    PyClass + pyo3::impl_::pyclass::PyClassBaseType<Initializer = PyClassInitializer<Self>>
{
    fn build_init() -> PyClassInitializer<Self>;
}

pub trait Init: PyTypeInfo + PyClass {
    fn init(self) -> PyClassInitializer<Self>;
    #[inline]
    fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        Bound::new(py, self.init())
    }
}
impl<T: PyClass<BaseType = I>, I: Abc> Init for T {
    #[inline]
    fn init(self) -> PyClassInitializer<Self> {
        I::build_init().add_subclass(self)
    }
}
