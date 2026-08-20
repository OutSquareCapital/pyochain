use crate::{
    abc,
    traits::{PyWrapper, PyoABC},
};
use either::Either;
use pyo3::{
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{PyInt, PyIterator, PyList, PyNotImplemented, PyTuple},
};
use pyo3_ext::{
    prelude::*,
    types::{FromCmp, PyCmpOut, PyDeque, PyIterable, PySupportsIndex},
};
use pyochain_macros::{try_cast, try_cast_into};
use tap::prelude::*;
#[pyclass(module = "pyochain.collections",frozen, generic, sequence, extends = abc::PyoMutableSequence)]
pub struct Deque(pub Py<PyDeque>);
#[pymethods]
impl Deque {
    #[new]
    #[pyo3(signature = (data=None, /, *elements, max_length=None))]
    fn new(
        py: Python<'_>,
        data: Option<Bound<'_, PyAny>>,
        elements: Bound<'_, PyTuple>,
        max_length: Option<Bound<'_, PyInt>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let deque = {
            try_cast_into! {
                match (data, elements.is_empty()) {
                    (None, _) => PyDeque::new(py, elements.into_any(), max_length)?,
                    (Some(Case::PyIterable(iterable)), true) => {
                        PyDeque::new(py, iterable.into_any(), max_length)?
                    }
                    (Some(any), true) => PyTuple::new(py, [any])
                        .map(Bound::into_any)
                        .and_then(|iterable| PyDeque::new(py, iterable, max_length))?,
                    (Some(any), false) => std::iter::once(any)
                        .chain(elements.into_iter())
                        .collect_bound::<PyList>(py)
                        .map(Bound::into_any)
                        .and_then(|x| PyDeque::new(py, x, max_length))?,
                }
            }
        };
        deque
            .unbind()
            .pipe(Self)
            .pipe(|slf| abc::PyoMutableSequence::build_init().add_subclass(slf))
            .pipe(Ok)
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDeque>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableSequence::build_init().add_subclass(Self(data.unbind()));
        Bound::new(py, initializer)
    }

    fn __repr__(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<String> {
        slf.get()
            .inner_bind(py)
            .repr()?
            .to_string()
            .replace("deque", &slf.get_type().name()?.to_string())
            .replace("maxlen", "max_length")
            .pipe(Ok)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner_bind(py).iter_py()
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.inner_bind(py).len()
    }

    fn __getitem__<'py>(&self, key: Bound<'py, PySupportsIndex>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(key.py()).as_any().get_item(key)
    }

    fn __setitem__(
        &self,
        key: Bound<'_, PySupportsIndex>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner_bind(key.py()).set_item(key, value)
    }

    fn __delitem__(&self, key: Bound<'_, PySupportsIndex>) -> PyResult<()> {
        self.inner_bind(key.py()).del_item(key)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }

    fn __iadd__(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let inner = slf.get().inner_bind(py);
        let other = if value.is(&slf) { inner } else { value };
        inner.iadd(other)?;
        Ok(())
    }

    fn __add__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner
                    .as_sequence()
                    .concat(d.get().inner_bind(py).as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref),
                Case::PyDeque(pyd) => inner
                    .as_sequence()
                    .concat(pyd.as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __mul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(value.py())
            .mul(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::from_ref)
    }
    fn __rmul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.__mul__(value)
    }
    fn __imul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<()> {
        self.inner_bind(value.py()).imul(value)?;
        Ok(())
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.lt(d.get().inner_bind(py)),
                Case::PyDeque(pyd) => inner.lt(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.le(d.get().inner_bind(py)),
                Case::PyDeque(pyd) => inner.le(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.gt(d.get().inner_bind(py)),
                Case::PyDeque(pyd) => inner.gt(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.ge(d.get().inner_bind(d.py())),
                Case::PyDeque(pyd) => inner.ge(pyd),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __eq__<'py>(&self, value: Bound<'py, PyAny>) -> PyCmpOut<'py, bool> {
        let py = value.py();
        let inner = self.inner_bind(py);
        try_cast! {
            match value {
                CaseExact::Deque(d) => inner.eq(d.get().inner_bind(py)).map(Either::Left),
                Case::PyDeque(pyd) => inner.eq(pyd).map(Either::Left),
                _ => PyNotImplemented::from_cmp(py),
            }
        }
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.inner_bind(py).reversed()
    }

    #[getter]
    fn max_length<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(py).getattr(intern!(py, "maxlen"))
    }

    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(x.py()).append(x)
    }

    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(x.py()).append_left(x)
    }

    fn extend(slf: Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner_bind(py);
        let other = if iterable.is(&slf) { inner } else { iterable };
        inner.extend(other)
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.inner_bind(py).clear()
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.inner_bind(py)
            .call_method0(intern!(py, "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::from_ref)
    }

    fn extend_left(slf: Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner_bind(py);
        let other = if iterable.is(&slf) { inner } else { iterable };
        inner.extend_left(other)
    }

    fn pop_left<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(py).call_method0(intern!(py, "popleft"))
    }
    #[pyo3(signature = (n=1))]
    fn rotate(slf: Bound<'_, Self>, n: isize) -> PyResult<Bound<'_, Self>> {
        slf.get().inner_bind(slf.py()).rotate(n).map(|_| slf)
    }

    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).insert(index, value)
    }
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.inner_bind(value.py()).count(value)
    }
    #[pyo3(signature = (x, start=None, stop=None, /))]
    fn index<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(x.py()).index(x, start, stop)
    }
    fn pop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(py).pop()
    }
    #[pyo3(signature = (value, /))]
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(value.py()).remove(value)
    }
}
