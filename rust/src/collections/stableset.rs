use crate::{
    abc::{self},
    display::get_repr,
    sets::SetMut,
    traits::{IntoPyochain, PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo, intern,
    prelude::*,
    types::{PyDict, PyIterator, PyNone, PyNotImplemented, PySet},
};
use pyo3_ext::{
    prelude::*,
    types::{PyAbstractSet, PyCmpOut},
};
use pyochain_macros::try_cast;
use tap::prelude::*;
#[pyclass(module = "pyochain._collections",frozen, generic, extends=abc::PyoMutableSet)]
pub struct StableSet(pub Py<PyDict>);
#[pymethods]
impl StableSet {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        PyDict::type_object(py)
            .call_method1(intern!(py, "fromkeys"), (data,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDict>() })
            .map(Bound::unbind)
            .map(|inner| abc::PyoMutableSet::build_init().add_subclass(Self(inner)))
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let name = slf.get_type().name()?;
        slf.get()
            .inner_bind(slf.py())
            .keys()
            .pipe(get_repr)
            .map(|repr| format!("{}({})", name, repr))
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).try_iter().unwrap()
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner_bind(slf.py()).len()
    }

    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(item.py()).contains(item)
    }

    fn __eq__<'py>(&self, other: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = other.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match other {
                PyAbstractSet(abc_set) => inner.keys_view().eq(abc_set).map(Either::Left),
                Self::exact(stable) => stable
                    .get()
                    .inner_bind(py)
                    .pipe(|set| inner.keys_view().eq(set))
                    .map(Either::Left),
                PySet(py_set) => inner.keys_view().eq(py_set).map(Either::Left),
                _ => PyNotImplemented::get(py)
                    .into_bound()
                    .pipe(Ok)
                    .map(Either::Right),
            }
        }
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableSet::build_init().add_subclass(Self(data.unbind()));
        Bound::new(py, initializer)
    }

    fn add(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner_bind(py).set_item(value, PyNone::get(py))
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner_bind(slf.py())
            .copy()
            .and_then(Self::from_ref)
    }

    fn discard(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).del_item(value)
    }

    fn intersection<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner_bind(py)
            .bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn union<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner_bind(py)
            .keys_view()
            .bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner_bind(py)
            .keys_view()
            .sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.inner_bind(py)
            .keys_view()
            .bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }
}
