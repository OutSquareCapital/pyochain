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
use pyo3_ext::types::PyDeque;
use pyochain_macros::py_abc;
use tap::Pipe;
pub trait PyWrapper: PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync {
    type Wrapped: PyTypeInfo + DerefToPyAny;
    fn inner(&self) -> &Py<Self::Wrapped>;
    fn inner_bind<'py>(&self, py: Python<'py>) -> &Bound<'py, Self::Wrapped> {
        self.inner().bind(py)
    }
    #[inline(always)]
    fn into_inner_bound<'py>(&self, py: Python<'py>) -> Bound<'py, Self::Wrapped> {
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
                let txt = format!(
                    "Input must be a '{}'' or a '{}', got '{}'",
                    wrapper, inner, incorrect
                );
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
/// Trait to convert a `Bound` of a Python type into a `Bound` of a PyoChain type, with the same underlying data.\
/// Useful for no-copy conversions, when the type is known at compile time.\
/// For example, this avoid checking the type of a `PyTuple` at runtime to convert it into a `Seq`.
pub trait IntoPyochain<'py, T: PyTypeInfo + IntoInit> {
    /// Convert a given Python type `T` (wrapped in a `Py` or `Bound`) into the corresponding pyochain wrapper type.\
    /// This will create the class and bind it to the python interpreter.\
    /// As such, prefer using `new` (or alike) methods if you simply want to use the struct in Rust.\
    /// That being said, prefer using the underlying Python type, as pyochain wrapper are, well, wrappers. There's not much to gain from using them in Rust.
    fn into_pyochain(self) -> PyResult<Bound<'py, T>>;
}
macro_rules! impl_into_pyochain {
    ($($py:ty => $pyochain:path),* $(,)?) => {
        $(
            impl<'py> IntoPyochain<'py, $pyochain> for Bound<'py, $py> {
                #[inline]
                fn into_pyochain(self) -> PyResult<Bound<'py, $pyochain>> {
                    Bound::new(self.py(), $pyochain(self.unbind()).init())
                }
            }
        )*
    };
}

impl_into_pyochain!(
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

    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(
        _args: &Bound<'_, PyTuple>,
        _kwargs: Option<&Bound<'_, PyDict>>,
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
                    iterable.into_pyochain()
                }
            }
        )*
    };
}
impl_flex_wrapper!(Set, SetMut, Iter, PyoVec, Seq, Dict);
