use crate::{
    abc::{self},
    pyo3_ext::{
        prelude::*,
        types::{PyDeque, PySupportsIndex},
    },
    traits::PyoABC,
};
use pyo3::{
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{PyInt, PyIterator, PyTuple},
};
use pyochain_macros::try_cast;
use tap::prelude::*;
#[pyclass(frozen, generic, sequence, extends = abc::PyoMutableSequence)]
pub struct Deque {
    #[pyo3(get)]
    inner: Py<PyDeque>,
}
#[pymethods]
impl Deque {
    #[new]
    #[pyo3(signature = (data=None, max_length=None))]
    fn new(
        py: Python<'_>,
        data: Option<Bound<'_, PyAny>>,
        max_length: Option<Bound<'_, PyInt>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        Ok(abc::PyoMutableSequence::build_init().add_subclass(Self {
            inner: PyDeque::new(
                py,
                data.unwrap_or_else(|| PyTuple::empty(py).into_any()),
                max_length,
            )?
            .unbind(),
        }))
    }

    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDeque>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableSequence::build_init().add_subclass(Self {
            inner: data.unbind(),
        });
        Bound::new(py, initializer)
    }

    fn __repr__(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<String> {
        slf.get()
            .inner
            .bind(py)
            .repr()?
            .to_string()
            .replace("deque", &slf.get_type().name()?.to_string())
            .replace("maxlen", "max_length")
            .pipe(Ok)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).try_iter().unwrap()
    }

    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.copy(py)
    }

    fn __len__(&self, py: Python<'_>) -> PyResult<usize> {
        self.inner.bind(py).len()
    }

    fn __getitem__<'py>(&self, key: Bound<'py, PySupportsIndex>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(key.py()).as_any().get_item(key)
    }

    fn __setitem__(
        &self,
        key: Bound<'_, PySupportsIndex>,
        value: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.inner.bind(key.py()).set_item(key, value)
    }

    fn __delitem__(&self, key: Bound<'_, PySupportsIndex>) -> PyResult<()> {
        self.inner.bind(key.py()).del_item(key)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }

    fn __iadd__(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        let inner = slf.get().inner.bind(py);
        let other = if value.is(&slf) { inner } else { value };
        inner.iadd(other)?;
        Ok(())
    }

    fn __add__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner
                    .as_sequence()
                    .concat(value.get().inner.bind(py).as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref),
                PyDeque => inner
                    .as_sequence()
                    .concat(value.as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __mul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(value.py())
            .mul(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::from_ref)
    }
    fn __rmul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        self.__mul__(value)
    }
    fn __imul__<'py>(&self, value: Bound<'py, PyInt>) -> PyResult<()> {
        self.inner.bind(value.py()).imul(value)?;
        Ok(())
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner.lt(value.get().inner.bind(py)),
                PyDeque => inner.lt(value),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner.le(value.get().inner.bind(py)),
                PyDeque => inner.le(value),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner.gt(value.get().inner.bind(py)),
                PyDeque => inner.gt(value),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner.ge(value.get().inner.bind(value.py())),
                PyDeque => inner.ge(value),
                _ => Err(PyTypeError::new_err("")),
            }
        }
    }

    fn __eq__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match value {
                Deque => inner.eq(value.get().inner.bind(py)),
                PyDeque => inner.eq(value),
                _ => Ok(false),
            }
        }
    }
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.inner.bind(py).reversed()
    }

    #[getter]
    fn max_length<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(py).getattr(intern!(py, "maxlen"))
    }

    fn append(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(x.py()).append(x)
    }

    fn append_left(&self, x: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(x.py()).append_left(x)
    }

    fn extend(slf: Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner.bind(py);
        let other = if iterable.is(&slf) { inner } else { iterable };
        inner.extend(other)
    }

    fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.bind(py).clear()
    }

    fn copy<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(py)
            .call_method0(intern!(py, "copy"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::from_ref)
    }

    fn extend_left(slf: Bound<'_, Self>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        let inner = slf.get().inner.bind(py);
        let other = if iterable.is(&slf) { inner } else { iterable };
        inner.extend_left(other)
    }

    fn pop_left<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(py).call_method0(intern!(py, "popleft"))
    }
    #[pyo3(signature = (n=1))]
    fn rotate(slf: Bound<'_, Self>, n: isize) -> PyResult<Bound<'_, Self>> {
        slf.get().inner.bind(slf.py()).rotate(n).map(|_| slf)
    }

    fn insert(&self, index: isize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).insert(index, value)
    }
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.inner.bind(value.py()).count(value)
    }
    #[pyo3(signature = (x, start=None, stop=None, /))]
    fn index<'py>(
        &self,
        x: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(x.py()).index(x, start, stop)
    }
    fn pop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(py).pop()
    }
    #[pyo3(signature = (value, /))]
    fn remove(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).remove(value)
    }
}
