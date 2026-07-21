use crate::{
    abc::{self, PyoABC},
    pyo3_ext::{
        prelude::*,
        pylibs,
        types::{PyAbstractSet, PyDeque, PySupportsIndex, PySupportsItems, pyitertools},
    },
    seq::{IntoPyochain, SetMut, get_repr},
};
use bound_from_any::{BoundFromAny, try_cast};
use either::Either;
use pyo3::{
    BoundObject, PyTypeInfo,
    exceptions::{PyKeyError, PyTypeError},
    intern,
    prelude::*,
    types::{
        PyDict, PyInt, PyIterator, PyList, PyMapping, PyNone, PyNotImplemented, PySet, PyTuple,
        PyType,
    },
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
#[derive(BoundFromAny)]
enum IntoUpdate<'py> {
    Dict(Bound<'py, PyDict>),
    Mapping(Bound<'py, PyMapping>),
    Iterable(Bound<'py, PyAny>),
}
type BoolOrNotImpl<'py> = Either<bool, Bound<'py, PyNotImplemented>>;
#[pyclass(frozen, generic, mapping, extends = abc::PyoMutableMapping)]
pub struct PyoCounter {
    pub inner: Py<PyDict>,
}
#[pymethods]
impl PyoCounter {
    #[new]
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn new(
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let inner = PyDict::new(py);

        update_counter(&inner, iterable, kwargs)?;
        Ok(abc::PyoMutableMapping::build_init().add_subclass(Self {
            inner: inner.unbind(),
        }))
    }
    #[staticmethod]
    fn from_ref(data: Bound<'_, PyDict>) -> PyResult<Bound<'_, Self>> {
        let py = data.py();
        let initializer = abc::PyoMutableMapping::build_init().add_subclass(Self {
            inner: data.unbind(),
        });
        Bound::new(py, initializer)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).try_iter().unwrap()
    }

    fn __len__<'py>(&self, py: Python<'py>) -> usize {
        self.inner.bind(py).len()
    }

    fn __getitem__<'py>(&self, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        match self.inner.bind(py).as_any().get_item(key) {
            Ok(value) => Ok(value),
            Err(err) => {
                if err.matches(py, PyKeyError::type_object(py))? {
                    Ok(PyInt::new(py, 0).into_any())
                } else {
                    Err(err)
                }
            }
        }
    }

    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyInt>) -> PyResult<()> {
        self.inner.bind(key.py()).set_item(key, value)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }
    #[allow(unused)]
    fn __missing__(&self, key: &Bound<'_, PyAny>) -> isize {
        0
    }
    #[pyo3(signature = (key, default=None, /))]
    fn get<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        self.inner
            .bind(key.py())
            .get_item(key)?
            .or(default)
            .pipe(Ok)
    }

    #[pyo3(signature = (key, default, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "setdefault"), (key, default))
    }

    fn total<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.inner
            .bind(py)
            .values()
            .try_iter()
            .unwrap()
            .pipe(|vals| pylibs::builtins::sum(&vals, &0))
    }
    #[pyo3(signature = (n=None))]
    fn most_common<'py>(
        &self,
        py: Python<'py>,
        n: Option<Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let items = self.inner.bind(py).items().try_iter().unwrap();
        let getter = pylibs::operator::itemgetter(py, 1)?;
        match n {
            None => pylibs::builtins::sorted_by(&items, true, &getter),
            Some(n) => {
                let kwargs = PyDict::new(py);
                kwargs.set_item(intern!(py, "key"), getter)?;
                py.import(intern!(py, "heapq"))?
                    .getattr(intern!(py, "nlargest"))?
                    .call((n, items), Some(&kwargs))
                    .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            }
        }
    }

    fn elements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        pylibs::itertools::chain::from_iterable(&pylibs::itertools::map_star(
            pyitertools::PyRepeat::type_object(py).into_any(),
            self.inner.bind(py).items().try_iter().unwrap(),
        )?)
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn update(
        &self,
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let inner = self.inner.bind(py);
        update_counter(&inner, iterable, kwargs)
    }
    #[pyo3(signature = (iterable=None, /, **kwargs))]
    fn subtract(
        &self,
        py: Python<'_>,
        iterable: Option<IntoUpdate<'_>>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<()> {
        let inner = self.inner.bind(py);
        subtract_counter(inner, iterable, kwargs)
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get_type()
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn __reduce__(slf: Bound<'_, Self>) -> (Bound<'_, PyType>, (Py<PyDict>,)) {
        let py = slf.py();
        return (Self::type_object(py), (slf.get().inner.clone_ref(py),));
    }

    fn __delitem__(&self, elem: Bound<'_, PyAny>) -> PyResult<()> {
        let inner = self.inner.bind(elem.py());
        if inner.contains(&elem)? {
            inner.del_item(elem)?;
        }
        Ok(())
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;

        if !&slf.is_truthy()? {
            Ok(format!("{}()", name))
        } else {
            let d_repr = slf
                .get()
                .most_common(py, None)
                // dict() preserves the ordering returned by most_common()
                .and_then(|x| x.as_any().pipe(PyDict::from_sequence))
                .or_else(|err| {
                    if err.is_instance_of::<PyTypeError>(py) {
                        // handle case where values are not orderable
                        slf.get().inner.clone_ref(py).into_bound(py).pipe(Ok)
                    } else {
                        Err(err)
                    }
                })?
                .repr()?;
            Ok(format!("{}({})", name, d_repr))
        }
    }

    fn __add__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = count.add(o.__getitem__(&elem)?)?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
            for (elem, count) in o.inner.bind(py).iter() {
                if !inner.contains(&elem)? && count.gt(0)? {
                    result.set_item(elem, count)?;
                }
            }
        }
        Self::from_ref(result)
    }

    fn __sub__<'py>(&self, py: Python<'py>, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let inner = self.inner.bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = count.sub(o.__getitem__(&elem)?)?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
        }
        for (elem, count) in o.inner.bind(py).iter() {
            if !inner.contains(&elem)? && count.lt(0)? {
                result.set_item(elem, PyInt::new(py, 0).sub(count)?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __or__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let result = PyDict::new(py);
        let o = other.get();
        let inner = self.inner.bind(py);
        for (elem, count) in inner.iter() {
            let other_count = o.__getitem__(&elem)?;
            let newcount = pylibs::builtins::max_of(py, (&count, &other_count))?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?
            }
        }
        for (elem, count) in other.get().inner.bind(other.py()).iter() {
            if !inner.contains(&elem)? && count.gt(0)? {
                result.set_item(elem, count)?
            }
        }
        Self::from_ref(result)
    }

    fn __and__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in self.inner.bind(py).iter() {
            let other_count = o.__getitem__(&elem)?;
            let newcount = pylibs::builtins::min_of(py, (&other_count, &count))?;
            if newcount.gt(0)? {
                result.set_item(elem, newcount)?;
            }
        }
        Self::from_ref(result)
    }

    fn __pos__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (elem, count) in self.inner.bind(py).iter() {
            if count.gt(0)? {
                result.set_item(elem, count)?
            }
        }
        Self::from_ref(result)
    }

    fn __neg__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, Self>> {
        let result = PyDict::new(py);
        for (elem, count) in self.inner.bind(py).iter() {
            if count.gt(0)? {
                result.set_item(elem, count.neg()?)?;
            }
        }
        Self::from_ref(result)
    }

    fn __iadd__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, count) = tup?;
            let new_count = self.__getitem__(&elem)?.add(count)?;
            inner.set_item(elem, new_count)?;
        }
        self._keep_positive(py)
    }

    fn __isub__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, count) = tup?;
            let new_count = self.__getitem__(&elem)?.sub(count)?;
            inner.set_item(elem, new_count)?;
        }
        self._keep_positive(py)
    }

    fn __ior__(&self, other: Bound<'_, PySupportsItems>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for tup in other.items()?.try_iter()?.map(extract_tup_from_item) {
            let (elem, other_count) = tup?;
            let count = self.__getitem__(&elem)?;
            if other_count.gt(count)? {
                inner.set_item(elem, other_count)?;
            }
        }
        self._keep_positive(py)
    }

    fn __iand__(&self, other: Bound<'_, PyMapping>) -> PyResult<()> {
        let py = other.py();
        let inner = self.inner.bind(py);
        for (elem, count) in inner.iter() {
            let other_count = other.as_any().get_item(&elem)?;
            if other_count.lt(count)? {
                inner.set_item(elem, other_count)?;
            }
        }
        self._keep_positive(py)
    }

    fn __eq__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<BoolOrNotImpl<'py>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        try_cast! {
            match other {
                PyoCounter => {
                    let o = other.get();
                    for c in [self, o] {
                        for elem in c.inner.bind(py).try_iter().unwrap() {
                            let e = elem?;
                            if self.__getitem__(&e)?.eq(o.__getitem__(&e)?)? {
                                continue;
                            } else {
                                return Ok(false).map(Either::Left);
                            }
                        }
                    }
                    Ok(true).map(Either::Left)

                },
                PyDict => inner.eq(other).map(Either::Left),
                _ => Ok(PyNotImplemented::get(py).into_bound()).map(Either::Right),
            }
        }
    }

    fn __ne__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<BoolOrNotImpl<'py>> {
        if !other.is_instance_of::<PyoCounter>() {
            PyNotImplemented::get(other.py())
                .into_bound()
                .pipe(Ok)
                .map(Either::Right)
        } else {
            self.__eq__(other)?.map_left(|x| !x).pipe(Ok)
        }
    }

    fn __le__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let o = other.get();
        for c in [self, o] {
            for elem in c.inner.bind(py).try_iter().unwrap() {
                let e = elem?;
                if self.__getitem__(&e)?.le(o.__getitem__(&e)?)? {
                    continue;
                } else {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn __lt__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<BoolOrNotImpl<'py>> {
        let is_ne = self.__ne__(other)?;
        let is_le = self.__le__(other)?;
        is_ne.map_left(|is_ne| is_le && is_ne).pipe(Ok)
    }

    fn __ge__(&self, other: &Bound<'_, Self>) -> PyResult<bool> {
        let py = other.py();
        let o = other.get();
        for c in [self, o] {
            for elem in c.inner.bind(py).try_iter().unwrap() {
                let e = elem?;
                if self.__getitem__(&e)?.ge(o.__getitem__(&e)?)? {
                    continue;
                } else {
                    return Ok(false);
                }
            }
        }

        Ok(true)
    }

    fn __gt__<'py>(&self, other: &Bound<'py, Self>) -> PyResult<BoolOrNotImpl<'py>> {
        let is_ge = self.__ge__(other)?;
        self.__ne__(&other)?
            .map_left(|is_ne| is_ge && is_ne)
            .pipe(Ok)
    }

    fn __xor__<'py>(&self, other: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let inner = self.inner.bind(py);
        let o = other.get();
        let result = PyDict::new(py);
        for (elem, count) in inner.iter() {
            let newcount = pylibs::builtins::abs(&count.sub(o.__getitem__(&elem)?)?)?;
            if newcount.is_truthy()? {
                result.set_item(elem, newcount)?
            }
        }
        for (elem, count) in other.get().inner.bind(py).iter() {
            if !inner.contains(&elem)? && count.is_truthy()? {
                result.set_item(elem, pylibs::builtins::abs(&count)?)?
            }
        }
        Self::from_ref(result)
    }

    fn __ixor__<'py>(&self, other: Bound<'py, Self>) -> PyResult<()> {
        let py = other.py();
        let o = other.get();
        let inner = self.inner.bind(py);
        for (elem, count) in inner.iter() {
            let new_item = pylibs::builtins::abs(&count.sub(o.__getitem__(&elem)?)?)?;
            inner.set_item(elem, new_item)?
        }
        for (elem, count) in other.get().inner.bind(py).iter() {
            if !inner.contains(&elem)? {
                inner.set_item(elem, pylibs::builtins::abs(&count)?)?
            }
        }
        self._keep_positive(py)
    }

    fn _keep_positive(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.bind(py);
        for (elem, count) in inner.iter() {
            if !(count.gt(0)?) {
                inner.del_item(elem)?;
            }
        }
        Ok(())
    }
}
#[inline(always)]
fn extract_tup_from_item(
    x: PyResult<Bound<'_, PyAny>>,
) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
    x?.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>()
}
#[inline]
fn update_counter(
    inner: &Bound<'_, PyDict>,
    iterable: Option<IntoUpdate<'_>>,
    kwargs: Option<Kwargs<'_>>,
) -> PyResult<()> {
    let py = inner.py();
    let zero = PyInt::new(py, 0).into_any();
    iterable
        .map(|iterable| match iterable {
            IntoUpdate::Dict(dict) => update_dict(inner, &dict, &zero),
            IntoUpdate::Mapping(mapping) => {
                if !inner.is_empty() {
                    for tup in mapping.items()?.try_iter()?.map(extract_tup_from_item) {
                        let (elem, count) = tup?;
                        let new_item =
                            count.add(inner.get_item(&elem)?.unwrap_or_else(|| zero.to_owned()))?;
                        inner.set_item(elem, new_item)?
                    }
                } else {
                    // fast path when counter is empty
                    inner.update(&mapping)?;
                }

                Ok(())
            }
            IntoUpdate::Iterable(it) => {
                for elem in it.try_iter()? {
                    let e = elem?;
                    let new_item = inner
                        .get_item(&e)?
                        .unwrap_or_else(|| zero.to_owned())
                        .add(1)?;
                    inner.set_item(&e, new_item)?;
                }

                Ok(())
            }
        })
        .transpose()?;

    kwargs
        .map(|kw| update_dict(inner, &kw, &zero))
        .transpose()?;
    Ok(())
}
#[inline]
fn update_dict(
    inner: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    zero: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if !inner.is_empty() {
        for (elem, count) in dict.iter() {
            let new_item = count.add(inner.get_item(&elem)?.unwrap_or_else(|| zero.to_owned()))?;
            inner.set_item(elem, new_item)?
        }
        Ok(())
    } else {
        // fast path when counter is empty
        inner.update(dict.as_mapping())
    }
}
#[inline]
fn subtract_counter(
    inner: &Bound<'_, PyDict>,
    iterable: Option<IntoUpdate<'_>>,
    kwargs: Option<Kwargs<'_>>,
) -> PyResult<()> {
    let py = inner.py();
    let zero = PyInt::new(py, 0).into_any();
    iterable
        .map(|it| match it {
            IntoUpdate::Dict(dict) => subtract_dict(inner, &dict, &zero),
            IntoUpdate::Mapping(mapping) => {
                for tup in mapping.items()?.try_iter()?.map(extract_tup_from_item) {
                    let (elem, count) = tup?;
                    let new_item = inner
                        .get_item(&elem)?
                        .unwrap_or_else(|| zero.to_owned())
                        .sub(count)?;
                    inner.set_item(elem, new_item)?
                }

                Ok(())
            }
            IntoUpdate::Iterable(it) => {
                for elem in it.try_iter()? {
                    let e = elem?;
                    let new_item = inner
                        .get_item(&e)?
                        .unwrap_or_else(|| zero.to_owned())
                        .sub(1)?;
                    inner.set_item(&e, new_item)?
                }
                Ok(())
            }
        })
        .transpose()?;
    kwargs
        .map(|dict| subtract_dict(inner, &dict, &zero))
        .transpose()?;
    Ok(())
}

#[inline]
fn subtract_dict(
    inner: &Bound<'_, PyDict>,
    dict: &Bound<'_, PyDict>,
    zero: &Bound<'_, PyAny>,
) -> PyResult<()> {
    for (elem, count) in dict.iter() {
        let new_item = inner
            .get_item(&elem)?
            .unwrap_or_else(|| zero.to_owned())
            .sub(count)?;
        inner.set_item(elem, new_item)?
    }
    Ok(())
}
