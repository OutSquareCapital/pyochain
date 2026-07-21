use crate::{
    abc::{self, PyoABC},
    pyo3_ext::{prelude::*, types::PyAbstractSet},
    seq::{IntoPyochain, SetMut, get_repr},
};
use bound_from_any::try_cast;
use pyo3::{
    PyTypeInfo, intern,
    prelude::*,
    types::{PyDict, PyIterator, PyNone, PySet},
};
use tap::prelude::*;
#[pyclass(frozen, generic, extends=abc::PyoMutableSet)]
pub struct StableSet {
    #[pyo3(get)]
    pub inner: Py<PyDict>,
}
#[pymethods]
impl StableSet {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        PyDict::type_object(py)
            .call_method1(intern!(py, "fromkeys"), (data,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDict>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoMutableSet::build_init().add_subclass(Self { inner }))
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?;
        slf.get()
            .inner
            .bind(slf.py())
            .keys()
            .as_sequence()
            .pipe(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner.bind(slf.py()).try_iter().unwrap()
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner.bind(slf.py()).len()
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(item.py()).contains(item)
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let inner = self.inner.bind(py);
        try_cast!(match other {
            PyAbstractSet => inner.keys_view().eq(other),
            StableSet => other
                .get()
                .inner
                .bind(py)
                .pipe(|set| inner.keys_view().eq(set)),
            PySet => inner.keys_view().eq(other),
            _ => Ok(false),
        })
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableSet::build_init().add_subclass(Self {
            inner: data.unbind(),
        });
        Bound::new(py, initializer)
    }

    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner.bind(py).set_item(value, PyNone::get(py))
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner
            .bind(slf.py())
            .copy()
            .and_then(Self::from_ref)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).del_item(value)
    }

    fn intersection<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner
            .bind(py)
            .bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn union<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner
            .bind(py)
            .keys_view()
            .bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner
            .bind(py)
            .keys_view()
            .sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner
            .bind(py)
            .keys_view()
            .bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }
}
