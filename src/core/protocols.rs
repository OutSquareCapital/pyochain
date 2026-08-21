use pyo3::{
    PyClass,
    prelude::*,
    types::{PyList, PySequence, PyTuple},
};
use pyo3_ext::{iter::TryCollectBoundIterator, types::PyIterable};
use pyochain_macros::{py_abc, try_cast_into};
use tap::Pipe;

use crate::{
    abc,
    core::{PyoVec, Seq},
    traits::{IntoPyochain, PyWrapper, PyoABC},
};
#[pyclass(module = "pyochain.core", frozen, generic)]
pub struct FlexibleInit;

#[py_abc(Seq, PyoVec)]
pub trait FlexInitProtocol: Sized + PyClass {
    #[pyo3(signature = (*elements))]
    #[new]
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>>;
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>>;
    #[staticmethod]
    #[pyo3(signature = (*elements))]
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>>;
}
#[py_abc(Seq, PyoVec)]
pub trait FlexWrapper: PyWrapper + FlexInitProtocol {
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>>;
}

impl FlexWrapper for PyoVec {
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>> {
        iterable.into_pyochain()
    }
}
impl FlexWrapper for Seq {
    fn wrap(iterable: Bound<'_, <Self as PyWrapper>::Wrapped>) -> PyResult<Bound<'_, Self>> {
        iterable.into_pyochain()
    }
}
impl FlexInitProtocol for Seq {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let tup = {
            match elements.len() {
                1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                    CaseExact::PyTuple(tuple) => tuple,
                    CaseExact::Self(inner) => inner.get().into_inner_bound(py),
                    Case::PySequence(sequence) => sequence.to_tuple()?,
                    Case::PyIterable(iterable) => iterable
                        .try_iter()?
                        .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?
                        .pipe(|x| PyTuple::new(py, x))?,
                    any => PyTuple::new(py, [any])?,
                }},
                _ => elements,
            }
        };
        tup.unbind()
            .pipe(Self)
            .pipe(|slf| abc::PyoSequence::build_init().add_subclass(slf))
            .pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::PyTuple(tuple) => tuple.into_pyochain(),
                CaseExact::Self(inner) => Ok(inner),
                Case::PySequence(sequence) => sequence.to_tuple()?.into_pyochain(),
                iterable => iterable
                    .try_iter()?
                    .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?
                    .pipe(|x| PyTuple::new(py, x))?
                    .into_pyochain(),
            }
        }
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.into_pyochain()
    }
}
impl FlexInitProtocol for PyoVec {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let list = match elements.len() {
            0 => PyList::empty(py),
            1 => try_cast_into! {match unsafe {elements.get_item_unchecked(0)} {
                CaseExact::Self(inner) => {
                    inner.get().into_inner_bound(py).as_sequence().to_list()?
                }
                Case::PySequence(sequence) => sequence.to_list()?,
                Case::PyIterable(iterable) => iterable.try_iter()?.try_collect_bound(py)?,
                any => PyList::new(py, [any])?,
            }},
            _ => elements.to_list(),
        };
        list.unbind()
            .pipe(Self)
            .pipe(|slf| abc::PyoMutableSequence::build_init().add_subclass(slf))
            .pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        let list = try_cast_into! {
            match iterable {
                CaseExact::PyList(list) => list.as_sequence().to_list()?,
                CaseExact::Self(inner) => {
                    inner.get().into_inner_bound(py).as_sequence().to_list()?
                }
                Case::PySequence(sequence) => sequence.to_list()?,
                iterable => iterable.try_iter()?.try_collect_bound(py)?,
            }
        };
        list.into_pyochain()
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.to_list().into_pyochain()
    }
}
