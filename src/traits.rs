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
pub trait IntoPyochain<'py, T: PyTypeInfo> {
    /// Convert a given Python type `T` (wrapped in a `Py` or `Bound`) into the corresponding pyochain wrapper type.\
    /// This will create the class and bind it to the python interpreter.\
    /// As such, prefer using `new` (or alike) methods if you simply want to use the struct in Rust.\
    /// That being said, prefer using the underlying Python type, as pyochain wrapper are, well, wrappers. There's not much to gain from using them in Rust.
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
impl<'py> IntoPyochain<'py, Range> for Bound<'py, PyRange> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Range>> {
        let py = self.py();
        let initializer = abc::PyoSequence::build_init().add_subclass(Range(self.unbind()));
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
impl<'py> IntoPyochain<'py, Iter> for Bound<'py, PyIterator> {
    #[inline]
    /// Convert a generic `PyIterator` into `Iter`, the generic wrapper for arbitrary iterators.
    fn into_pyochain(self) -> PyResult<Bound<'py, Iter>> {
        let py = self.py();
        let initializer = abc::PyoIterator::build_init().add_subclass(Iter(self.unbind()));
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyochain<'py, collections::Deque> for Bound<'py, PyDeque> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, collections::Deque>> {
        let py = self.py();
        let initializer =
            abc::PyoMutableSequence::build_init().add_subclass(collections::Deque(self.unbind()));
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
