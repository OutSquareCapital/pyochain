use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyRange, PySet, PyTuple},
};

use crate::{abc, collections, dict, mixins::Checkable, pyovec, range, seq, sets};
pub trait PyWrapper<T: PyTypeInfo>:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn as_inner(&self) -> &Py<T>;
    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(value: &'r Bound<'py, PyAny>) -> PyResult<&'r Bound<'py, T>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().as_inner().bind(py))
            .or_else(|_| value.cast_exact::<T>())
            .map_err(|_| {
                let py = value.py();
                let wrapper_name = Self::type_object(py).name().unwrap();
                let inner_name = T::type_object(py).name().unwrap();
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
    ($($wrapper:ty => $T:ty),* $(,)?) => {
        $(
            impl PyWrapper<$T> for $wrapper {

                #[inline]
                fn as_inner(&self) -> &Py<$T> {
                    &self.inner
                }
            }
        )*
    };
}
impl_py_wrapper! {
    seq::Seq => PyTuple,
    pyovec::PyoVec => PyList,
    sets::Set => PyFrozenSet,
    sets::SetMut => PySet,
    range::Range => PyRange,
    dict::Dict => PyDict,
    collections::StableSet => PyDict,
    collections::PyoCounter => PyDict,
    collections::HeapMin => PyList,
    collections::HeapMax => PyList,
}

/// Trait to convert a `Bound` of a Python type into a `Bound` of a PyoChain type, with the same underlying data.\
/// Useful for no-copy conversions, when the type is known at compile time.\
/// For example, this avoid checking the type of a `PyTuple` at runtime to convert it into a `Seq`.
pub trait IntoPyochain<'py, T: PyTypeInfo> {
    fn into_pyochain(self) -> PyResult<Bound<'py, T>>;
}

impl<'py> IntoPyochain<'py, seq::Seq> for Bound<'py, PyTuple> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, seq::Seq>> {
        let py = self.py();
        let initializer = abc::PyoSequence::build_init().add_subclass(seq::Seq {
            inner: self.unbind(),
        });
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, pyovec::PyoVec> for Bound<'py, PyList> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, pyovec::PyoVec>> {
        let py = self.py();
        let initializer = abc::PyoMutableSequence::build_init().add_subclass(pyovec::PyoVec {
            inner: self.unbind(),
        });
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, sets::Set> for Bound<'py, PyFrozenSet> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, sets::Set>> {
        let py = self.py();
        let initializer = abc::PyoSet::build_init().add_subclass(sets::Set {
            inner: self.unbind(),
        });
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, sets::SetMut> for Bound<'py, PySet> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, sets::SetMut>> {
        let py = self.py();
        let initializer = abc::PyoMutableSet::build_init().add_subclass(sets::SetMut {
            inner: self.unbind(),
        });
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, dict::Dict> for Bound<'py, PyDict> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, dict::Dict>> {
        let py = self.py();
        let initializer = abc::PyoMutableMapping::build_init().add_subclass(dict::Dict {
            inner: self.unbind(),
        });
        Bound::new(py, initializer)
    }
}

pub trait PyoABC: PyTypeInfo + PyClass {
    fn build_init() -> PyClassInitializer<Self>;
}

impl PyoABC for abc::PyoIterable {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
}
impl PyoABC for abc::PyoIterator {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoIterable::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoCollection {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoIterable::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoSequence {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoSet {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoMutableSet {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoSet::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoMutableSequence {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoSequence::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoMapping {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for abc::PyoMutableMapping {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoMapping::build_init().add_subclass(Self)
    }
}
impl PyoABC for crate::collections::Heap {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoMutableSequence::build_init().add_subclass(Self)
    }
}
