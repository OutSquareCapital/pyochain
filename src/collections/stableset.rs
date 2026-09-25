use crate::{
    abc::{self},
    core::SetMut,
    traits::{FlexWrapper, PyWrapper},
};
use derive_more::{Deref, From};
use either::Either;
use pyo3::{
    prelude::*,
    types::{PyDict, PyIterator, PyNone, PyNotImplemented, PySet},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyAbstractSet, PyCmpOut},
};
use pyochain_macros::try_cast;
use tap::prelude::*;
#[derive(From, Deref)]
#[pyclass(module = "pyochain.collections",frozen, generic, extends=abc::PyoMutableSet)]
pub struct StableSet(Py<PyDict>);
#[pymethods]
impl StableSet {
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.bind(py).keys().pipe(|repr| Self::get_repr(&repr))
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.bind(py).iter_py()
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.bind(py).len()
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        self.bind(item.py()).contains(item)
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = other.py();
        let inner = self.bind(py);
        try_cast! {
            match other {
                Case::PyAbstractSet(abc_set) => inner.keys_view().eq(abc_set).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }

    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.bind(py).set_item(value, PyNone::get(py))
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.bind(py).copy().and_then(Self::wrap)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.bind(value.py()).del_item(value)
    }

    fn intersection<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.bind(py)
            .bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .try_into_py()
    }

    fn union<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.bind(py)
            .keys_view()
            .bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .try_into_py()
    }

    fn difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.bind(py)
            .keys_view()
            .sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .try_into_py()
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.bind(py)
            .keys_view()
            .bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .try_into_py()
    }
}
