use crate::{
    abc::{self, PyoABC},
    pyo3_ext::{
        prelude::*,
        pylibs,
        types::{PyAbstractSet, PyDeque, PySupportsIndex, PySupportsItems, pyitertools},
    },
    seq::{IntoPyochain, SetMut, get_repr},
    try_cast,
};

use pyo3::{
    PyTypeInfo,
    exceptions::{PyKeyError, PyNotImplementedError, PyTypeError},
    intern,
    prelude::*,
    types::{PyDict, PyInt, PyIterator, PyList, PyMapping, PyNone, PySet, PyTuple, PyType},
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
            PyAbstractSet => {
                inner.keys_view().eq(other)
            }
            StableSet => {
                other
                    .get()
                    .inner
                    .bind(py)
                    .pipe(|set| inner.keys_view().eq(set))
            }
            PySet => {
                inner.keys_view().eq(other)
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
    fn count<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        self.inner.bind(value.py()).count(value)
    }
    #[pyo3(signature = (x, start=None, stop=None, /))]
    fn index<'py>(
        &self,
        x: Bound<'py, PyAny>,
        start: Option<Bound<'py, PyAny>>,
        stop: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyInt>> {
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

#[pyclass(frozen, generic, extends=abc::PyoMutableMapping)]
pub struct PyoCounter {
    pub inner: Py<PyDict>,
}
impl PyoCounter {
    fn iter_items<'py>(
        &self,
        py: Python<'py>,
    ) -> impl Iterator<Item = (Bound<'py, PyAny>, Bound<'py, PyInt>)> {
        self.inner
            .bind(py)
            .iter()
            .map(|(k, v)| unsafe { (k, v.cast_into_unchecked::<PyInt>()) })
    }
}
#[pymethods]
impl PyoCounter {
    #[new]
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let data = PyDict::new(py);
        iterable.map(|it| update_counter(&data, &it)).transpose()?;
        kwargs
            .map(|kw| update_counter_from_dict(&data, &kw))
            .transpose()?;
        let init = abc::PyoMutableMapping::build_init().add_subclass(Self {
            inner: data.unbind(),
        });
        Ok(init)
    }
    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let init = abc::PyoMutableMapping::build_init().add_subclass(Self {
            inner: data.unbind(),
        });
        Bound::new(py, init)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).try_iter().unwrap()
    }

    fn __len__(&self, py: Python<'_>) -> usize {
        self.inner.bind(py).len()
    }

    fn __getitem__<'py>(&self, key: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = key.py();
        match self.inner.bind(key.py()).as_any().get_item(&key) {
            Ok(value) => Ok(unsafe { value.cast_into_unchecked::<PyInt>() }),
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(py) {
                    return Ok(self.__missing__(key));
                } else {
                    return Err(err);
                }
            }
        }
    }

    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: Bound<'_, PyInt>) -> PyResult<()> {
        self.inner.bind(key.py()).set_item(key, value)
    }

    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }

    fn __missing__<'py>(&self, key: Bound<'py, PyAny>) -> Bound<'py, PyInt> {
        PyInt::new(key.py(), 0)
    }
    #[pyo3(signature = (key, default=None, /))]
    fn get<'py>(
        &self,
        key: Bound<'py, PyInt>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .bind(key.py())
            .get_item(key)?
            .or(default)
            .pipe(Ok)
    }
    #[pyo3(signature = (key, default=None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "setdefault"), (key, default))
    }

    fn total<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .bind(py)
            .values_view()
            .as_any()
            .try_iter()
            .unwrap()
            .pipe(|x| pylibs::builtins::sum(&x, &0))
    }
    #[pyo3(signature = (n=None))]
    fn most_common<'py>(&self, py: Python<'py>, n: Option<isize>) -> PyResult<Bound<'py, PyList>> {
        let items = self.inner.bind(py).items_view().try_iter().unwrap();
        let key = pylibs::operator::itemgetter(py, 1)?;
        match n {
            None => pylibs::builtins::sorted_by(&items, true, &key),

            _ => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "key"), key)?;
                py.import("heapq")?
                    .getattr("nlargest")?
                    .call((n, items), Some(&kwargs))
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            }
        }
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.inner
            .bind(py)
            .items_view()
            .try_iter()
            .unwrap()
            .pipe(|x| {
                pylibs::itertools::map_star(pyitertools::PyRepeat::type_object(py).into_any(), x)
            })
            .and_then(|x| pylibs::itertools::chain::from_iterable(&x))
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let data = self.inner.bind(py);
        iterable.map(|it| update_counter(&data, &it)).transpose()?;
        kwargs.map(|kw| update_counter(&data, &kw)).transpose()?;
        Ok(())
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn subtract(
        &self,
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let inner = self.inner.bind(py);
        iterable.map(|it| update_counter(&inner, &it)).transpose()?;
        kwargs
            .map(|kw| update_counter_from_dict(&inner, &kw))
            .transpose()?;
        Ok(())
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        Self::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn __reduce__(slf: Bound<'_, Self>) -> (Bound<'_, PyType>, (Py<PyDict>,)) {
        let py = slf.py();
        return (Self::type_object(py), (slf.get().inner.clone_ref(py),));
    }

    fn __delitem__(&self, elem: Bound<'_, PyAny>) -> PyResult<()> {
        let py = elem.py();
        let inner = self.inner.bind(py);
        if inner.contains(&elem)? {
            inner.del_item(elem)?
        }
        Ok(())
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        if !slf.is_truthy()? {
            slf.get_type().name().map(|name| format!("{}()", name))
        } else {
            // dict() preserves the ordering returned by most_common()
            let d = match slf
                .get()
                .most_common(py, None)?
                .as_any()
                .pipe(PyDict::from_sequence)
            {
                Ok(d) => d,
                Err(err) => {
                    if err.is_instance_of::<PyTypeError>(py) {
                        PyDict::new(py)
                    } else {
                        return Err(err);
                    }
                }
            }
            .repr()?
            .to_string();
            slf.get_type().name().map(|name| format!("{}({})", name, d))
        }
    }

    fn __add__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        let result = PyDict::new(py);
        for (k, count) in self.iter_items(py) {
            let newcount = count.add(other.get_item(&k)?)?;
            if newcount.gt(0)? {
                result.set_item(k, newcount)?;
            }
        }
        for (k, count) in other.get().iter_items(py) {
            if !inner.contains(&k)? && count.gt(0)? {
                result.set_item(&k, count)?;
            }
        }
        Self::from_ref(result)
    }

    fn __sub__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        let result = PyDict::new(py);
        for (k, count) in self.iter_items(other.py()) {
            let newcount = count.sub(other.get_item(&k)?)?;
            if newcount.gt(0)? {
                result.set_item(k, newcount)?;
            }
        }
        for (k, count) in other.get().iter_items(py) {
            if !inner.contains(&k)? && count.lt(0)? {
                result.set_item(k, count.neg()?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __or__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        let result = PyDict::new(py);
        for (k, count) in self.iter_items(other.py()) {
            let other_count = other.get_item(&k)?;
            let newcount = count
                .extract::<usize>()?
                .max(other_count.extract::<usize>()?);
            if newcount > 0 {
                result.set_item(k, newcount)?;
            }
        }
        for (k, count) in other.get().iter_items(py) {
            if !inner.contains(&k)? && count.gt(0)? {
                result.set_item(k, count)?;
            }
        }
        Self::from_ref(result)
    }

    fn __and__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let result = PyDict::new(py);
        for (k, count) in self.inner.bind(other.py()).iter() {
            let other_count = other.get_item(&k)?;
            let newcount = other_count
                .extract::<usize>()?
                .min(count.extract::<usize>()?);
            if newcount > 0 {
                result.set_item(k, newcount)?;
            }
        }
        Self::from_ref(result)
    }

    fn __pos__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (elem, count) in self.iter_items(py) {
            if count.gt(0)? {
                result.set_item(elem, count)?;
            }
        }
        Self::from_ref(result)
    }

    fn __neg__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (k, count) in self.iter_items(py) {
            if count.lt(0)? {
                result.set_item(k, count.neg()?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __iadd__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let inner = self.inner.bind(other.py());
        other.items()?.try_iter()?.try_for_each(|t| {
            let tuple = t?.cast_into_exact::<PyTuple>()?;
            let (k, count) = { (tuple.get_item(0)?, tuple.get_item(1)?) };
            inner.as_any().get_item(&k)?.iadd(count).map(|_| ())
        })?;
        keep_positive(inner)
    }

    fn __isub__<'py>(&self, other: Bound<'py, PySupportsItems>) -> PyResult<()> {
        let inner = self.inner.bind(other.py());
        other.items()?.try_iter()?.try_for_each(|t| {
            let tuple = t?.cast_into_exact::<PyTuple>()?;
            let (k, count) = { (tuple.get_item(0)?, tuple.get_item(1)?) };
            inner.as_any().get_item(&k)?.isub(count).map(|_| ())
        })?;
        keep_positive(inner)
    }

    fn __ior__<'py>(&self, other: Bound<'py, PySupportsItems>) -> PyResult<()> {
        let inner = self.inner.bind(other.py());
        other.items()?.try_iter()?.try_for_each(|t| {
            let tuple = t?.cast_into_exact::<PyTuple>()?;
            let (k, other_count) = { (tuple.get_item(0)?, tuple.get_item(1)?) };
            let count = inner.get_item(&k)?;
            if other_count.gt(&count)? {
                inner.set_item(k, other_count)?;
            }
            Ok::<(), PyErr>(())
        })?;
        keep_positive(inner)
    }

    fn __iand__(&self, other: Bound<'_, PyMapping>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for (k, count) in self.iter_items(py) {
            let other_count = other.get_item(&k)?.pipe(|x| x.cast_into_exact::<PyInt>())?;
            if other_count.lt(&count)? {
                inner.as_any().set_item(k, other_count)?
            }
        }
        keep_positive(inner)
    }

    fn __ixor__(&self, other: &Bound<'_, Self>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for (k, v) in self.iter_items(py) {
            let new = v.sub(other.get_item(&k)?)?.abs()?;
            inner.set_item(k, new)?;
        }
        for (k, count) in other.get().iter_items(py) {
            if !inner.contains(&k)? {
                inner.as_any().set_item(k, count.abs()?)?;
            }
        }
        keep_positive(inner)
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let inner = self.inner.bind(py).as_any();
        try_cast!(match other {
            PyoCounter => {
                for c in [inner, other.get().inner.bind(py)] {
                    for e in c.try_iter()? {
                        let e = e?;
                        if inner.get_item(&e)?.ne(other.get_item(e)?)? {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            }
            PyDict => {
                inner.eq(other)
            }
            _ => {
                Err(PyNotImplementedError::new_err(""))
            }
        })
    }

    fn __ne__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(!(self.__eq__(other)?))
    }

    fn __le__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let inner = self.inner.bind(py).as_any();
        for c in [inner.as_any(), other.get().inner.bind(py).as_any()] {
            for e in c.try_iter()? {
                let e = e?;
                if inner.get_item(&e)?.gt(&other.get_item(e)?)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn __lt__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(self.__le__(other)? && self.__ne__(other)?)
    }

    fn __ge__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let inner = self.inner.bind(py).as_any();
        for c in [inner, other.get().inner.bind(py).as_any()] {
            for e in c.try_iter()? {
                let e = e?;
                if inner.get_item(&e)?.lt(&other.get_item(e)?)? {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn __gt__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        Ok(self.__ge__(other)? && self.__ne__(other)?)
    }

    fn __xor__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other_inner = other.get().inner.bind(py).as_any();
        let result = PyDict::new(py);
        for (elem, count) in self.iter_items(py) {
            let newcount = count.sub(other_inner.get_item(&elem)?)?.abs()?;
            if newcount.is_truthy()? {
                result.set_item(elem, newcount)?;
            };
        }
        for (elem, count) in other.get().iter_items(py) {
            if !self.__contains__(&elem)? && count.is_truthy()? {
                result.set_item(elem, count.abs()?)?;
            }
        }
        Self::from_ref(result)
    }
}

fn keep_positive(data: &Bound<'_, PyDict>) -> PyResult<()> {
    data.iter().try_for_each(|(elem, count)| {
        if count.lt(0)? {
            data.del_item(&elem)?;
        }
        Ok(())
    })
}
#[inline]
fn update_counter_from_dict(
    data: &Bound<'_, PyDict>,
    iterable: &Bound<'_, PyDict>,
) -> PyResult<()> {
    let py = data.py();
    if !data.is_empty() {
        iterable.iter().try_for_each(|(k, count)| {
            let new_item = count.add(data.get_item(&k)?.unwrap_or(PyInt::new(py, 0).into_any()))?;
            data.set_item(k, new_item)
        })
    } else {
        // fast path when counter is empty
        data.update(iterable.as_mapping())
    }
}

#[inline]
fn update_counter(data: &Bound<'_, PyDict>, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
    let py = data.py();
    try_cast!(match iterable {
        PyDict => {
            update_counter_from_dict(data, iterable)
        }
        PyMapping => {
            if !data.is_empty() {
                iterable
                    .items()?
                    .iter()
                    .map(|x| x.cast_into_exact::<PyTuple>())
                    .try_for_each(|x| {
                        let res = x?;
                        let e = res.get_item(0)?;
                        let count = res.get_item(1)?;
                        let new_item = count
                            .add(data.get_item(&e)?.unwrap_or(PyInt::new(py, 0).into_any()))?;
                        data.set_item(e, new_item)
                    })
            } else {
                // fast path when counter is empty
                data.update(iterable)
            }
        }
        _ => {
            iterable.try_iter()?.try_for_each(|elem| {
                let e = elem?;
                let new_item = data
                    .get_item(&e)?
                    .unwrap_or(PyInt::new(py, 0).into_any())
                    .add(1)?;
                data.set_item(e, new_item)
            })
        }
    })
}
