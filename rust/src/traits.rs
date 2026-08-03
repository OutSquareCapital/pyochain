use pyo3::{
    PyClass, PyTypeInfo,
    exceptions::PyTypeError,
    prelude::*,
    types::{
        DerefToPyAny, PyDict, PyFrozenSet, PyIterator, PyList, PyRange, PySequence, PySet, PyTuple,
    },
};

use crate::{
    abc, collections,
    dict::Dict,
    iterators::Iter,
    pyo3_ext::types::PyDeque,
    pyovec::PyoVec,
    range::Range,
    seq::Seq,
    sets::{Set, SetMut},
    sliceview::SliceView,
};
pub trait PyWrapper<T: PyTypeInfo + DerefToPyAny>:
    PyClass<Frozen = pyo3::pyclass::boolean_struct::True> + Sync
{
    fn inner(&self) -> &Py<T>;
    fn inner_bind<'py>(&self, py: Python<'py>) -> &Bound<'py, T> {
        self.inner().bind(py)
    }
    #[inline(always)]
    fn into_inner_bound<'py>(&self, py: Python<'py>) -> Bound<'py, T> {
        self.inner().clone_ref(py).into_bound(py)
    }

    /// Extracts the inner type of `Self` from an arbitrary python object.\
    /// For example, if `Self` is `seq::Seq`, this will extract the inner `PyTuple` from a `seq::Seq` or a `PyTuple`.
    #[inline]
    fn extract_union<'py, 'r>(value: &'r Bound<'py, PyAny>) -> PyResult<&'r Bound<'py, T>> {
        let py = value.py();
        value
            .cast_exact::<Self>()
            .map(|x| x.get().inner_bind(py))
            .or_else(|_| value.cast_exact::<T>())
            .map_err(|_| {
                let py = value.py();
                let wrapper = Self::type_object(py).name().unwrap();
                let inner = T::type_object(py).name().unwrap();
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
            impl PyWrapper<$T> for $wrapper {

                #[inline(always)]
                fn inner(&self) -> &Py<$T> {
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
impl PyWrapper<PySequence> for SliceView {
    #[inline(always)]
    fn inner(&self) -> &Py<PySequence> {
        &self.inner
    }
}
/// Trait to convert a `Bound` of a Python type into a `Bound` of a PyoChain type, with the same underlying data.\
/// Useful for no-copy conversions, when the type is known at compile time.\
/// For example, this avoid checking the type of a `PyTuple` at runtime to convert it into a `Seq`.
pub trait IntoPyochain<'py, T: PyTypeInfo> {
    fn into_pyochain(self) -> PyResult<Bound<'py, T>>;
}

impl<'py> IntoPyochain<'py, Seq> for Bound<'py, PyTuple> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Seq>> {
        let py = self.py();
        let initializer = abc::PyoSequence::build_init().add_subclass(Seq(self.unbind()));
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, PyoVec> for Bound<'py, PyList> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, PyoVec>> {
        let py = self.py();
        let initializer = abc::PyoMutableSequence::build_init().add_subclass(PyoVec(self.unbind()));
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, Set> for Bound<'py, PyFrozenSet> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Set>> {
        let py = self.py();
        let initializer = abc::PyoSet::build_init().add_subclass(Set(self.unbind()));
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, SetMut> for Bound<'py, PySet> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, SetMut>> {
        let py = self.py();
        let initializer = abc::PyoMutableSet::build_init().add_subclass(SetMut(self.unbind()));
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, Dict> for Bound<'py, PyDict> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Dict>> {
        let py = self.py();
        let initializer = abc::PyoMutableMapping::build_init().add_subclass(Dict(self.unbind()));
        Bound::new(py, initializer)
    }
}

pub trait PyoABC: PyTypeInfo + PyClass {
    fn build_init() -> PyClassInitializer<Self>;
}

impl PyoABC for abc::PyoIterable {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(abc::Checkable).add_subclass(Self)
    }
}

impl PyoABC for abc::PyoSized {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(abc::Checkable).add_subclass(Self)
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
impl PyoABC for collections::Heap {
    fn build_init() -> PyClassInitializer<Self> {
        abc::PyoMutableSequence::build_init().add_subclass(Self)
    }
}
