use std::ops::Deref;

use crate::{abc, collections, core};
use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{self, DerefToPyAny},
};
use pyo3_ext::{prelude::*, types::PyDeque};
use pyochain_macros::py_abc;
use tap::prelude::*;
pub trait PyWrapper:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync + Deref<Target = Py<Self::Wrapped>>
{
    type Wrapped: PyTypeInfo + DerefToPyAny;
    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<&'r Bound<'py, Self::Wrapped>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().bind(py))
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

    fn get_repr(obj: &Bound<'_, PyAny>) -> PyResult<String> {
        let name = Self::type_object(obj.py()).name()?;
        match obj.len() {
            Ok(0) => Ok(format!("{name}()")),
            Ok(_) => {
                let mut elements = obj.repr()?.to_string();
                elements.pop();
                elements.remove(0);
                Ok(format!("{name}({elements})"))
            }
            Err(err) => Err(err),
        }
    }
}
/// Implement `PyWrapper` for pyochain types in one line.
macro_rules! impl_py_wrapper {
    ($($wrapper:ty => $T:ty),* $(,)?) => {
        $(
            impl PyWrapper for $wrapper {
                type Wrapped = $T;
            }
        )*
    };
}
impl_py_wrapper! {
    core::Seq => types::PyTuple,
    core::PyoVec => types::PyList,
    core::Set => types::PyFrozenSet,
    core::SetMut => types::PySet,
    core::Range => types::PyRange,
    core::Dict => types::PyDict,
    core::iterators::Iter => types::PyIterator,
    collections::StableSet => types::PyDict,
    collections::PyoCounter => types::PyDict,
    collections::HeapMin => types::PyList,
    collections::HeapMax => types::PyList,
    collections::Deque => PyDeque,
    core::SliceView => types::PySequence,
}
macro_rules! impl_try_from_py {
    ($($py:ty => $pyochain:path),* $(,)?) => {
        $(
            impl TryFromPy<$py> for $pyochain {
                #[inline]
                fn try_from_py(obj: Bound<'_, $py>) -> PyResult<Bound<'_, Self>> {
                    Bound::new(obj.py(), obj.unbind().conv::<Self>().init())
                }
            }
        )*
    };
}

impl_try_from_py!(
    types::PyTuple => core::Seq,
    types::PyList => core::PyoVec,
    types::PyFrozenSet => core::Set,
    types::PySet => core::SetMut,
    types::PyRange => core::Range,
    types::PyDict => core::Dict,
    types::PyIterator => core::iterators::Iter,
    PyDeque => collections::Deque,
    types::PyDict => collections::StableSet,
    types::PyDict => collections::PyoCounter,

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
        args: &Bound<'_, types::PyTuple>,
        kwargs: Option<&Bound<'_, types::PyDict>>,
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
    core::Seq,
    core::PyoVec,
    core::Set,
    core::SetMut,
    collections::StableSet,
    core::iterators::Iter,
    core::Dict,
    collections::PyoCounter
)]
pub trait FlexWrapper: PyWrapper + TryFromPy<Self::Wrapped> {
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>> {
        iterable.try_into_py()
    }
}
