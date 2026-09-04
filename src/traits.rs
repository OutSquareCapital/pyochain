use crate::{
    abc, collections,
    core::{Dict, PyoVec, Range, Seq, Set, SetMut, SliceView, iterators::Iter},
};
use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{
        DerefToPyAny, PyDict, PyFrozenSet, PyIterator, PyList, PyRange, PySequence, PySet, PyTuple,
    },
};
use pyo3_ext::prelude::TryFromPy;
use pyo3_ext::{prelude::*, types::PyDeque};
use pyochain_macros::py_abc;
use tap::Pipe;
pub trait PyWrapper: PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync {
    type Wrapped: PyTypeInfo + DerefToPyAny;
    fn inner(&self) -> &Py<Self::Wrapped>;
    fn inner_bind<'py>(&self, py: Python<'py>) -> &Bound<'py, Self::Wrapped> {
        self.inner().bind(py)
    }
    #[inline(always)]
    fn inner_into_bound<'py>(&self, py: Python<'py>) -> Bound<'py, Self::Wrapped> {
        self.inner().clone_ref(py).into_bound(py)
    }
    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<&'r Bound<'py, Self::Wrapped>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().inner_bind(py))
            .or_else(|_| value.cast_exact::<Self::Wrapped>())
            .map_err(|_| {
                let py = value.py();
                let wrapper = Self::type_object(py).name().unwrap();
                let inner = Self::Wrapped::type_object(py).name().unwrap();
                let incorrect = value.get_type().name().unwrap();
                let txt = format!("Input must be a '{wrapper}'' or a '{inner}', got '{incorrect}'");
                PyTypeError::new_err(txt)
            })
    }
}
/// Implement `PyWrapper` for pyochain types in one line.
macro_rules! impl_py_wrapper {
    ($($wrapper:ty => $T:ty),* $(,)?) => {
        $(
            impl PyWrapper for $wrapper {
                type Wrapped = $T;

                #[inline(always)]
                fn inner(&self) -> &Py<Self::Wrapped> {
                    &self.0
                }
            }
        )*
    };
}
impl_py_wrapper! {
    Seq => PyTuple,
    PyoVec => PyList,
    Set => PyFrozenSet,
    SetMut => PySet,
    Range => PyRange,
    Dict => PyDict,
    Iter => PyIterator,
    collections::StableSet => PyDict,
    collections::PyoCounter => PyDict,
    collections::HeapMin => PyList,
    collections::HeapMax => PyList,
    collections::Deque => PyDeque,
}
/// Named struct so need to implement `PyWrapper` manually.
impl PyWrapper for SliceView {
    type Wrapped = PySequence;
    #[inline(always)]
    fn inner(&self) -> &Py<Self::Wrapped> {
        &self.inner
    }
}
macro_rules! impl_try_from_py {
    ($($py:ty => $pyochain:path),* $(,)?) => {
        $(
            impl TryFromPy<$py> for $pyochain {
                #[inline]
                fn try_from_py(obj: Bound<'_, $py>) -> PyResult<Bound<'_, Self>> {
                    Bound::new(obj.py(), Self(obj.unbind()).init())
                }
            }
        )*
    };
}

impl_try_from_py!(
    PyTuple => Seq,
    PyList => PyoVec,
    PyFrozenSet => Set,
    PySet => SetMut,
    PyRange => Range,
    PyDict => Dict,
    PyIterator => Iter,
    PyDeque => collections::Deque

);
#[py_abc(
    abc::PyoIterable,
    abc::PyoSized,
    abc::PyoIterator,
    abc::PyoCollection,
    abc::PyoSequence,
    abc::PyoSet,
    abc::PyoMutableSet,
    abc::PyoMutableSequence,
    abc::PyoMapping,
    abc::PyoMutableMapping,
    abc::PyoReversible
)]
trait PyoABC:
    PyTypeInfo + PyClass + pyo3::impl_::pyclass::PyClassBaseType<Initializer = PyClassInitializer<Self>>
{
    #[skip]
    fn build_init() -> PyClassInitializer<Self>;

    #[pyo3(signature = (*args, **kwargs))]
    #[new]
    #[allow(unused_variables)]
    fn new(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyClassInitializer<Self> {
        Self::build_init()
    }
}
macro_rules! impl_pyoabc {
    ($($base:ty => $sub:ty),* $(,)?) => {
        $(
            impl PyoABC for $sub {
                fn build_init() -> PyClassInitializer<Self> {
                    <$base as PyoABC>::build_init().add_subclass(Self)
                }
            }
        )*
    };
}

impl_pyoabc! {
    abc::Checkable => abc::PyoSized,
    abc::Checkable => abc::PyoIterable,
    abc::PyoIterable => abc::PyoReversible,
    abc::PyoIterable => abc::PyoIterator,
    abc::PyoIterable => abc::PyoCollection,
    abc::PyoCollection => abc::PyoSequence,
    abc::PyoCollection => abc::PyoSet,
    abc::PyoSet => abc::PyoMutableSet,
    abc::PyoSequence => abc::PyoMutableSequence,
    abc::PyoCollection => abc::PyoMapping,
    abc::PyoMapping => abc::PyoMutableMapping,
    abc::PyoMutableSequence => collections::Heap,
}
impl PyoABC for abc::Checkable {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(abc::Checkable)
    }
}
pub trait IntoInit: PyTypeInfo + PyClass {
    fn init(self) -> PyClassInitializer<Self>;

    fn into_bound(self, py: Python<'_>) -> PyResult<Bound<'_, Self>> {
        Bound::new(py, self.init())
    }
}

impl<
    T: PyClass<BaseType = I>,
    I: PyoABC + pyo3::impl_::pyclass::PyClassBaseType<Initializer = PyClassInitializer<I>>,
> IntoInit for T
{
    fn init(self) -> PyClassInitializer<Self> {
        I::build_init().add_subclass(self)
    }
}

#[py_abc(
    Seq,
    PyoVec,
    Set,
    SetMut,
    collections::StableSet,
    Iter,
    Dict,
    collections::PyoCounter
)]
pub trait FlexWrapper: PyWrapper {
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>>;
}
impl FlexWrapper for collections::StableSet {
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        iterable.unbind().pipe(Self).into_bound(py)
    }
}
impl FlexWrapper for collections::PyoCounter {
    fn wrap(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        data.unbind().pipe(Self).into_bound(py)
    }
}
macro_rules! impl_flex_wrapper {
    ($($ty:ty),*) => {
        $(
            impl FlexWrapper for $ty {
                fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>> {
                    iterable.try_into_py()
                }
            }
        )*
    };
}
impl_flex_wrapper!(Set, SetMut, Iter, PyoVec, Seq, Dict);
#[allow(unused)]
pub trait OptionExt<T, R, E> {
    fn map_transpose(self, func: impl FnOnce(T) -> R) -> Result<Option<R>, E>;
    fn and_then_transpose(self, func: impl FnOnce(T) -> Result<R, E>) -> Result<Option<R>, E>;
}
impl<T, R, E> OptionExt<T, R, E> for Option<Result<T, E>> {
    /// Transforms an `Option<Result<T, E>>` into a `Result<Option<R>, E>` by applying a function to the `Ok` value if it exists.\
    /// If the `Option` is `None`, it returns `Ok(None)`.\
    /// If the `Option` is `Some(Err(e))`, it returns `Err(e)`.\
    /// If the `Option` is `Some(Ok(item))`, it applies the function to `item` and wraps the result in `Some`.\
    /// This can be useful to replace nested `transpose()` and `map()` calls with a single method call, improving readability.
    #[inline]
    fn map_transpose(self, func: impl FnOnce(T) -> R) -> Result<Option<R>, E> {
        match self {
            Some(Ok(item)) => Ok(Some(func(item))),
            None => Ok(None),
            Some(Err(e)) => Err(e),
        }
    }
    /// Similar to `map_transpose`, but the function returns a `Result<R, E>`.\
    /// Allows for chaining operations that may fail, while still handling the `Option` case.
    #[inline]
    fn and_then_transpose(self, func: impl FnOnce(T) -> Result<R, E>) -> Result<Option<R>, E> {
        match self {
            Some(Ok(item)) => func(item).map(Some),
            None => Ok(None),
            Some(Err(e)) => Err(e),
        }
    }
}
