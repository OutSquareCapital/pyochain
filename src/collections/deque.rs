use crate::{abc, traits::IntoInit};
use derive_more::{Deref, From};
use either::Either;
use pyo3::{
    PyTypeInfo,
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{PyInt, PyIterator, PyNotImplemented, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut, PyDeque, PyIterable, PySupportsIndex},
};
use pyochain_macros::{try_cast, try_cast_into};
use tap::prelude::*;
#[derive(From, Deref)]
#[pyclass(module = "pyochain.collections",frozen, generic, sequence, extends = abc::PyoMutableSequence)]
pub struct Deque(Py<PyDeque>);
#[pymethods]
impl Deque {
    #[new]
    #[pyo3(signature = (*elements, max_length=None))]
    fn new(
        elements: Bound<'_, PyTuple>,
        max_length: Option<Bound<'_, PyInt>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        match elements.len() {
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                Case::PyIterable(iterable) => PyDeque::new(iterable.into_any(), max_length)?,
                any => tuple!(any)
                    .map(Bound::into_any)
                    .and_then(|iterable| PyDeque::new(iterable, max_length))?,
            }},
            _ => PyDeque::new(elements.into_any(), max_length)?,
        }
        .unbind()
        .conv::<Self>()
        .init()
        .pipe(Ok)
    }

    #[pyo3(signature = (iterable, /, max_length=None))]
    #[staticmethod]
    fn from_iter<'py>(
        iterable: Bound<'py, PyAny>,
        max_length: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        PyDeque::new(iterable, max_length)?.try_into_py()
    }
    #[staticmethod]
    #[pyo3(signature = (*elements, max_length=None))]
    fn of<'py>(
        elements: Bound<'py, PyTuple>,
        max_length: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        PyDeque::new(elements.into_any(), max_length)?.try_into_py()
    }
    #[pyo3(signature = (data, /))]
    #[staticmethod]
    fn wrap(data: Bound<'_, PyDeque>) -> PyResult<Bound<'_, Self>> {
        data.try_into_py()
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        self.bind(py)
            .repr()?
            .to_string()
            .replace("deque", &Self::type_object(py).name()?.to_string())
            .replace("maxlen", "max_length")
            .pipe(Ok)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.bind(py).iter_py()
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.bind(py).len()
    }

    fn __getitem__<'py>(&self, key: Bound<'py, PySupportsIndex>) -> PyResult<Bound<'py, PyAny>> {
        self.bind(key.py()).as_any().get_item(key)
    }

    fn __setitem__(
        &self,
        key: Bound<'_, PySupportsIndex>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.bind(key.py()).set_item(key, value)
    }

    fn __delitem__(&self, key: Bound<'_, PySupportsIndex>) -> PyResult<()> {
        self.bind(key.py()).del_item(key)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.bind(key.py()).contains(key)
    }

    fn __iadd__(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let inner = slf.get().bind(py);
        let other = if value.is(slf) { inner } else { value };
        inner.iadd(other)?;
        Ok(())
    }

    fn __add__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner
                    .as_sequence()
                    .concat(d.get().bind(py).as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::wrap),
                Case::PyDeque(pyd) => inner
                    .as_sequence()
                    .concat(pyd.as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::wrap),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __mul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.bind(value.py())
            .mul(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::wrap)
    }
    fn __rmul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.__mul__(value)
    }
    fn __imul__(&self, value: Bound<'_, PyInt>) -> PyResult<()> {
        self.bind(value.py()).imul(value)?;
        Ok(())
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.lt(d.get().bind(py)),
                Case::PyDeque(pyd) => inner.lt(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.le(d.get().bind(py)),
                Case::PyDeque(pyd) => inner.le(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.gt(d.get().bind(py)),
                Case::PyDeque(pyd) => inner.gt(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.ge(d.get().bind(d.py())),
                Case::PyDeque(pyd) => inner.ge(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __eq__<'py>(&self, value: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = value.py();
        let inner = self.bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.eq(d.get().bind(py)).map(Either::Left),
                Case::PyDeque(pyd) => inner.eq(pyd).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.bind(py).reversed()
    }

    #[getter]
    fn max_length<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.bind(py).getattr(intern!(py, "maxlen"))
    }

    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.bind(x.py()).append(x)
    }

    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.bind(x.py()).append_left(x)
    }

    fn extend(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().bind(py);
        let other = if iterable.is(slf) { inner } else { iterable };
        inner.extend(other)
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.bind(py).clear()
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.bind(py)
            .call_method0(intern!(py, "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::wrap)
    }

    fn extend_left(slf: &Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().bind(py);
        let other = if iterable.is(slf) { inner } else { iterable };
        inner.extend_left(other)
    }

    fn pop_left<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.bind(py).call_method0(intern!(py, "popleft"))
    }
    #[pyo3(signature = (n=1))]
    fn rotate(slf: Bound<'_, Self>, n: isize) -> PyResult<Bound<'_, Self>> {
        slf.get().bind(slf.py()).rotate(n).map(|()| slf)
    }

    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.bind(value.py()).insert(index, value)
    }
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.bind(value.py()).count(value)
    }
    #[pyo3(signature = (x, start=None, stop=None, /))]
    fn index<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.bind(x.py()).index(x, start, stop)
    }
    fn pop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.bind(py).pop()
    }
    #[pyo3(signature = (value, /))]
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.bind(value.py()).remove(value)
    }
}
