use pyo3::{
    prelude::*,
    types::{PyList, PySequence, PyTuple},
};
use pyo3_ext::iter::TryCollectBoundIterator;
use pyochain_macros::{py_abc, try_cast_into};
use tap::Pipe;

use crate::{
    core::{PyoVec, Seq},
    traits::{IntoPyochain, PyWrapper},
};
#[pyclass(module = "pyochain.core", frozen, generic)]
pub struct FlexibleInit;

#[py_abc(Seq, PyoVec)]
pub trait FlexInitProtocol: Sized {
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
