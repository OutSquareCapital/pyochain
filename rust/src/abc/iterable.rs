use crate::abc::{
    PyoCollection, PyoIterator, PyoMapping, PyoMutableMapping, PyoMutableSequence, PyoMutableSet,
    PyoSequence, PyoSet,
};
use crate::mixins::Checkable;
use crate::pyo3_ext::prelude::*;
use crate::tools;
use pyo3::{PyClass, PyTypeInfo, prelude::*};
use tap::prelude::*;

pub trait PyoABC: PyTypeInfo + PyClass {
    fn build_init() -> PyClassInitializer<Self>;
}

impl PyoABC for PyoIterable {
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
}
impl PyoABC for PyoIterator {
    fn build_init() -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoCollection {
    fn build_init() -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoSequence {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoSet {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableSet {
    fn build_init() -> PyClassInitializer<Self> {
        PyoSet::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableSequence {
    fn build_init() -> PyClassInitializer<Self> {
        PyoSequence::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMapping {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableMapping {
    fn build_init() -> PyClassInitializer<Self> {
        PyoMapping::build_init().add_subclass(Self)
    }
}
impl PyoABC for crate::collections::Heap {
    fn build_init() -> PyClassInitializer<Self> {
        PyoMutableSequence::build_init().add_subclass(Self)
    }
}
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Py<tools::Iter>> {
        slf.into_any().pipe(tools::Iter::new)
    }
}
