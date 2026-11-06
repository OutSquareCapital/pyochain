use std::ops::Deref;

use crate::{abc, collections, core, mixins};
use pyo3::{
    PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{self, DerefToPyAny},
};
use pyo3_ext::{prelude::*, types::PyDeque};
use pyochain_macros::py_abc;
use tap::prelude::*;
pub trait PyWrapper: PyFrozenClass + Deref<Target = Py<Self::Wrapped>> {
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
trait PyoABC: Abc + Init {
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
            impl Abc for $sub {
                fn build_init() -> PyClassInitializer<Self> {
                    <$base as Abc>::build_init().add_subclass(Self)
                }
            }
        )*
    };
}

impl_pyoabc! {
    mixins::Checkable => abc::PyoSized,
    mixins::Checkable => abc::PyoIterable,
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
impl Abc for mixins::Checkable {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(mixins::Checkable)
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
