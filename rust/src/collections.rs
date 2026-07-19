use crate::{
    abc::{self, PyoABC},
    pyo3_ext::{
        prelude::*,
        types::{PyAbstractSet, PyDeque, PySupportsIndex},
    },
    seq::{IntoPyochain, SetMut, get_repr},
    try_cast,
};

use pyo3::{
    PyTypeInfo,
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{PyDict, PyInt, PyIterator, PyNone, PySet},
};
use tap::prelude::*;
#[pyclass(frozen, generic, extends=abc::PyoMutableSet)]
pub struct StableSet {
    #[pyo3(get)]
    inner: Py<PyDict>,
}
impl StableSet {
    fn py_keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner.bind(py).call_method0(intern!(py, "keys"))
    }
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
        try_cast!(match other {
            PyAbstractSet => {
                self.py_keys(py)?.eq(other)
            }
            StableSet => {
                other
                    .get()
                    .inner
                    .bind(py)
                    .pipe(|set| self.py_keys(py)?.eq(set))
            }
            PySet => {
                self.py_keys(py)?.eq(other)
            }
            _ => {
                Ok(false)
            }
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
        self.py_keys(py)?
            .bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn union<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.py_keys(py)?
            .bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.py_keys(py)?
            .sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }

    fn symmetric_difference<'py>(&self, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        let py = other.py();
        self.py_keys(py)?
            .bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })?
            .into_pyochain()
    }
}
#[pyclass(frozen, generic, sequence, extends = abc::PyoMutableSequence)]
pub struct Deque {
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
            inner: PyDeque::new(py, data, max_length)?.unbind(),
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
        self.inner.bind(key.py()).get_item(key)
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

    fn __iadd__(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).iadd(value)?;
        Ok(())
    }

    fn __add__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner
                    .as_sequence()
                    .concat(value.get().inner.bind(py).as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref)
            }
            PyDeque => {
                inner
                    .as_sequence()
                    .concat(value.as_sequence())
                    .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
                    .and_then(Self::from_ref)
            }
            _ => {
                Err(PyTypeError::new_err(""))
            }
        })
    }

    fn __mul__<'py>(&self, py: Python<'py>, value: usize) -> PyResult<Bound<'py, Self>> {
        self.inner
            .bind(py)
            .as_sequence()
            .repeat(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })
            .and_then(Self::from_ref)
    }

    fn __imul__<'py>(&self, py: Python<'py>, value: usize) -> PyResult<()> {
        self.inner
            .bind(py)
            .as_sequence()
            .in_place_repeat(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyDeque>() })?;
        Ok(())
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner.lt(value.get().inner.bind(py))
            }
            PyDeque => {
                inner.lt(value)
            }
            _ => {
                Err(PyTypeError::new_err(""))
            }
        })
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner.le(value.get().inner.bind(py))
            }
            PyDeque => {
                inner.le(value)
            }
            _ => {
                Err(PyTypeError::new_err(""))
            }
        })
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner.gt(value.get().inner.bind(py))
            }
            PyDeque => {
                inner.gt(value)
            }
            _ => {
                Err(PyTypeError::new_err(""))
            }
        })
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner.ge(value.get().inner.bind(value.py()))
            }
            PyDeque => {
                inner.ge(value)
            }
            _ => {
                Err(PyTypeError::new_err(""))
            }
        })
    }

    fn __eq__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let inner = self.inner.bind(py);
        try_cast!(match value {
            Deque => {
                inner.eq(value.get().inner.bind(py))
            }
            PyDeque => {
                inner.eq(value)
            }
            _ => {
                Ok(false)
            }
        })
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

    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(iterable.py()).extend(iterable)
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

    fn extend_left(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(iterable.py()).extend_left(iterable)
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
}
