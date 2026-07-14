use crate::{abc, mixins, pylibs};
use either::Either;
use pyo3::exceptions::PyTypeError;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyDict, PyInt, PyIterator, PyList, PyRange, PyRangeMethods, PySequence, PyString, PyTuple,
};
use pyo3::{PyTypeInfo, ffi, intern, prelude::*};
use tap::Pipe;

trait PyWrapper: PyTypeInfo {
    type Inner: PyTypeInfo;
    /// Extract the type of a value and check if it is one of two types, returning an `Either` with the result.\
    /// Returns a `PyErr` if the value is not one of the two types.
    #[inline]
    fn extract_union<'py, 'r>(
        value: &'r Bound<'py, PyAny>,
    ) -> PyResult<Either<&'r Bound<'py, Self>, &'r Bound<'py, Self::Inner>>> {
        value
            .cast_exact::<Self>()
            .map(Either::Left)
            .or_else(|_| value.cast_exact::<Self::Inner>().map(Either::Right))
            .map_err(|_| {
                let py = value.py();
                let wrapper_name = Self::type_object(py).name().unwrap();
                let inner_name = Self::Inner::type_object(py).name().unwrap();
                let value_name = value.get_type().name().unwrap();
                let txt = format!(
                    "Input must be a '{}'' or a '{}', got '{}'",
                    wrapper_name, inner_name, value_name
                );
                PyTypeError::new_err(txt)
            })
    }
}
impl PyWrapper for Vec {
    type Inner = PyList;
}

impl PyWrapper for Seq {
    type Inner = PyTuple;
}

/// Trait to convert a `Bound` of a Python type into a `Bound` of a PyoChain type, with the same underlying data.\
/// Useful for no-copy conversions, when the type is known at compile time.\
/// For example, this avoid checking the type of a `PyTuple` at runtime to convert it into a `Seq`.
trait IntoPyoChain<'py, T: PyTypeInfo> {
    fn into_pyochain(self) -> PyResult<Bound<'py, T>>;
}

impl<'py> IntoPyoChain<'py, Seq> for Bound<'py, PyTuple> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Seq>> {
        let py = self.py();
        let initializer = PyClassInitializer::from(mixins::Checkable)
            .add_subclass(abc::PyoIterable)
            .add_subclass(abc::PyoCollection)
            .add_subclass(abc::PyoSequence)
            .add_subclass(Seq {
                inner: self.unbind(),
            });
        Bound::new(py, initializer)
    }
}
impl<'py> IntoPyoChain<'py, Vec> for Bound<'py, PyList> {
    #[inline]
    fn into_pyochain(self) -> PyResult<Bound<'py, Vec>> {
        let py = self.py();
        let initializer = PyClassInitializer::from(mixins::Checkable)
            .add_subclass(abc::PyoIterable)
            .add_subclass(abc::PyoCollection)
            .add_subclass(abc::PyoSequence)
            .add_subclass(abc::PyoMutableSequence)
            .add_subclass(Vec {
                inner: self.unbind(),
            });
        Bound::new(py, initializer)
    }
}

#[pyclass(frozen, generic, sequence, extends=abc::PyoSequence)]
pub struct Seq {
    #[pyo3(get)]
    inner: Py<PyTuple>,
}
#[pymethods]
impl Seq {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.cast_exact::<Self>()
            .map(|x| x.get().inner.clone_ref(py))
            .or_else(|_| unsafe {
                PyTuple::type_object(py)
                    .call1((&data,))?
                    .cast_into_unchecked::<PyTuple>()
                    .unbind()
                    .pipe(Ok)
            })
            .map(|inner| {
                PyClassInitializer::from(mixins::Checkable)
                    .add_subclass(abc::PyoIterable)
                    .add_subclass(abc::PyoCollection)
                    .add_subclass(abc::PyoSequence)
                    .add_subclass(Self { inner })
            })
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name().unwrap();
        let repr = slf
            .get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .into_sequence()
            .pipe_ref(get_repr)?;
        format!("{}({})", name, repr).pipe(Ok)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.bind(py).try_iter().unwrap()
    }

    fn __len__(&self, py: Python) -> usize {
        self.inner.bind(py).len()
    }

    fn __getitem__<'py>(&self, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        self.inner.bind(py).as_any().get_item(index)
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> bool {
        let py = other.py();
        let left = self.inner.bind(py);
        if let Ok(o) = other.cast_exact::<Self>() {
            left.eq(o.get().inner.bind(py)).unwrap()
        } else if let Ok(o) = other.cast_exact::<PyTuple>() {
            left.eq(o).unwrap()
        } else {
            false
        }
    }

    fn __hash__(slf: Bound<'_, Self>) -> isize {
        let py = slf.py();
        slf.get().inner.clone_ref(py).bind(py).hash().unwrap()
    }

    fn __reversed__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        let py = slf.py();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }
    fn __lt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).lt(Self::extract_union(value)?)
    }
    fn __le__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).le(Self::extract_union(value)?)
    }
    fn __gt__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).gt(Self::extract_union(value)?)
    }
    fn __ge__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).ge(Self::extract_union(value)?)
    }
    fn __add__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(value)
    }
    fn __mul__<'py>(slf: Bound<'_, Self>, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __rmul__<'py>(
        slf: Bound<'py, Self>,
        value: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __repeat__<'py>(slf: Bound<'py, Self>, count: isize) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&PyInt::new(slf.py(), count))
    }
    fn __concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(other)
    }
    fn __inplace_concat__<'py>(
        &self,
        other: &Bound<'py, PySequence>,
    ) -> PyResult<Bound<'py, PySequence>> {
        let py = other.py();
        let tup = Self::extract_union(other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner()
            .as_sequence();
        self.inner.bind(py).as_sequence().in_place_concat(tup)
    }
    fn __inplace_repeat__<'py>(
        slf: Bound<'py, Self>,
        count: isize,
    ) -> PyResult<Bound<'py, PySequence>> {
        slf.get()
            .inner
            .bind(slf.py())
            .as_sequence()
            .in_place_repeat(count as usize)
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        py_count(&self.inner.bind(value.py()).as_any(), &value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py_index(&self.inner.bind(value.py()).as_any(), value, start, stop)
    }

    fn repeat<'py>(&self, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "__mul__"), (n,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
            .and_then(Bound::into_pyochain)
    }
    fn concat<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other_seq = Self::extract_union(other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner()
            .as_sequence();
        self.inner
            .bind(py)
            .as_sequence()
            .concat(other_seq)?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
            .into_pyochain()
    }
}

#[pyclass(frozen, sequence, extends=abc::PyoSequence)]
pub struct Range {
    #[pyo3(get)]
    inner: Py<PyRange>,
}

#[pymethods]
impl Range {
    #[pyo3(signature = (start, stop, step = 1))]
    #[new]
    fn new(
        start: Bound<'_, PyInt>,
        stop: Bound<'_, PyInt>,
        step: isize,
    ) -> PyResult<PyClassInitializer<Self>> {
        PyRange::type_object(start.py())
            .call1((start, stop, step))?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyRange>() })
            .unbind()
            .pipe(|inner| {
                PyClassInitializer::from(mixins::Checkable)
                    .add_subclass(abc::PyoIterable)
                    .add_subclass(abc::PyoCollection)
                    .add_subclass(abc::PyoSequence)
                    .add_subclass(Self { inner })
            })
            .pipe(Ok)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.clone_ref(py).bind(py).try_iter().unwrap()
    }

    fn __len__(&self, py: Python) -> usize {
        self.inner
            .bind(py)
            .pipe(|x| unsafe { x.cast_unchecked::<PySequence>() })
            .len()
            .unwrap()
    }

    fn __getitem__<'py>(&self, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        self.inner.bind(py).get_item(index)
    }

    fn __repr__(slf: Bound<'_, Self>) -> String {
        let py = slf.py();
        let name = slf.get_type().name().unwrap();
        let inner = slf.get().inner.bind(py);

        let params = format!(
            "{}, {}, {}",
            inner.start().unwrap(),
            inner.stop().unwrap(),
            inner.step().unwrap()
        );
        format!("{}({})", name, params)
    }

    fn __eq__(&self, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(value.py()).eq(value)
    }
    fn __hash__(slf: Bound<'_, Self>) -> PyResult<isize> {
        slf.get().inner.bind(slf.py()).hash()
    }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }
    fn __reversed__<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyIterator> {
        let py = slf.py();
        slf.get()
            .inner
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyInt>> {
        let py = value.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "count"), (&value,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyInt>() })
    }
    fn index<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = value.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "index"), (&value,))
    }
}
#[pyclass(frozen, generic, sequence, extends=abc::PyoMutableSequence)]
pub struct Vec {
    #[pyo3(get)]
    inner: Py<PyList>,
}
#[pymethods]
impl Vec {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        data.pipe(|x| PyList::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .map(Bound::unbind)
            .map(|inner| {
                PyClassInitializer::from(mixins::Checkable)
                    .add_subclass(abc::PyoIterable)
                    .add_subclass(abc::PyoCollection)
                    .add_subclass(abc::PyoSequence)
                    .add_subclass(abc::PyoMutableSequence)
                    .add_subclass(Self { inner })
            })
    }
    #[staticmethod]
    fn from_ref<'py>(data: Bound<'py, PyList>) -> PyResult<Bound<'py, Self>> {
        data.into_pyochain()
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner.bind(slf.py()).try_iter().unwrap()
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner.bind(key.py()).contains(key)
    }
    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        Ok(format!(
            "{}({})",
            name,
            slf.get().inner.bind(py).as_sequence().pipe(get_repr)?
        ))
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner.bind(slf.py()).len()
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = other.py();
        let o = Self::extract_union(&other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).eq(&o).or(Ok(false))
    }
    fn __reversed__<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyIterator> {
        let py = slf.py();
        slf.get()
            .inner
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }

    fn __add__<'py>(slf: Bound<'py, Self>, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        Self::extract_union(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner()
            .as_sequence()
            .pipe(|x| slf.get().inner.bind(py).as_sequence().concat(x))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }

    fn __iadd__<'py>(&self, value: &Bound<'py, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "__iadd__"), (value,))?;
        Ok(())
    }
    fn __inplace_concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<()> {
        self.__iadd__(other)
    }

    fn __concat__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        self.concat(other)
    }
    fn __mul__<'py>(slf: Bound<'_, Self>, value: Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __rmul__<'py>(
        slf: Bound<'py, Self>,
        value: Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&value)
    }
    fn __repeat__<'py>(slf: Bound<'py, Self>, count: isize) -> PyResult<Bound<'py, Self>> {
        slf.get().repeat(&PyInt::new(slf.py(), count))
    }
    fn __inplace_repeat__<'py>(slf: Bound<'py, Self>, count: isize) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        Self::repeat_mut(slf, &PyInt::new(py, count))
    }
    fn __imul__(slf: Bound<'_, Self>, value: usize) -> PyResult<()> {
        slf.get()
            .inner
            .bind(slf.py())
            .as_sequence()
            .in_place_repeat(value)?;
        Ok(())
    }

    fn __gt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).gt(other)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).ge(other)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).lt(other)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = Self::extract_union(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).le(other)
    }

    fn __getitem__<'py>(
        slf: Bound<'py, Self>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.get().inner.bind(slf.py()).as_any().get_item(index)
    }
    fn __setitem__(
        slf: Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        slf.get().inner.bind(key.py()).as_any().set_item(key, value)
    }

    fn __delitem__(slf: Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.get().inner.bind(key.py()).as_any().del_item(key)
    }

    #[pyo3(signature = (*, reverse=false))]
    fn sort(slf: Bound<'_, Self>, reverse: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        slf.get()
            .inner
            .bind(py)
            .call_method(intern!(py, "sort"), (), Some(&kwargs))?;
        Ok(slf)
    }
    #[pyo3(signature = (key, *, reverse=false))]
    fn sort_by<'py>(
        slf: Bound<'py, Self>,
        key: Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let py = key.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item(intern!(py, "key"), key)?;
        kwargs.set_item(intern!(py, "reverse"), reverse)?;
        slf.get()
            .inner
            .bind(py)
            .call_method(intern!(py, "sort"), (), Some(&kwargs))?;
        Ok(slf)
    }

    fn append(&self, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).append(value)
    }

    fn extend(&self, iterable: Bound<'_, PyAny>) -> PyResult<()> {
        let py = iterable.py();
        iterable
            .try_iter()
            .map(|_| unsafe { ffi::PyList_Extend(self.inner.as_ptr(), iterable.as_ptr()) })
            .and_then(|x| {
                if x == 0 {
                    Ok(())
                } else {
                    Err(PyErr::fetch(py))
                }
            })
    }

    fn repeat<'py>(&self, n: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        self.inner
            .bind(py)
            .call_method1(intern!(py, "__mul__"), (n,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }
    fn repeat_mut<'py>(slf: Bound<'py, Self>, n: &Bound<'_, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = n.py();
        slf.get()
            .inner
            .bind(py)
            .call_method1(intern!(py, "__imul__"), (n,))?;
        Ok(slf)
    }

    fn insert(&self, index: usize, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.bind(value.py()).insert(index, value)
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        slf.get()
            .inner
            .bind(py)
            .call_method0(intern!(py, "clear"))?;
        Ok(())
    }

    fn reverse(slf: Bound<'_, Self>) -> PyResult<()> {
        slf.get().inner.bind(slf.py()).reverse()
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get_type()
            .call1((slf.get().inner.bind(slf.py()),))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (value, /))]
    fn count<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        py_count(&self.inner.bind(value.py()).as_any(), &value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        start: Option<&Bound<'py, PyAny>>,
        stop: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        py_index(&self.inner.bind(value.py()).as_any(), value, start, stop)
    }

    fn concat<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        Self::extract_union(&other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner()
            .pipe(|other| {
                self.inner
                    .bind(py)
                    .as_sequence()
                    .concat(other.as_sequence())
            })
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Bound::into_pyochain)
    }

    fn concat_mut<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other = Self::extract_union(&other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner()
            .as_sequence();
        slf.get()
            .inner
            .bind(py)
            .as_sequence()
            .in_place_concat(other)
            .map(|_| slf)
    }
}

fn get_repr<'py>(obj: &Bound<'py, PySequence>) -> PyResult<Bound<'py, PyString>> {
    static PFORMAT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    let py = obj.py();
    let length = obj.len()?;

    match length {
        0 => Ok(PyString::new(py, "")),
        _ => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("sort_dicts", false).unwrap();
            PFORMAT
                .import(py, "pprint", "pformat")?
                .call((obj,), Some(&kwargs))
                .map(|x| unsafe { x.cast_into_unchecked::<PyString>() })
                .map(|x| {
                    let full = x.to_str().unwrap();
                    PyString::new(py, &full[1..full.len() - 1])
                })
        }
    }
}
/// Helper function to call the `index` method on a `list` or `tuple` object with optional start and stop parameters.
#[inline]
fn py_index<'py>(
    obj: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
    start: Option<&Bound<'py, PyAny>>,
    stop: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    let method_name = intern!(py, "index");
    match (start, stop) {
        (Some(start), Some(stop)) => obj.call_method1(method_name, (value, start, stop)),
        (Some(start), None) => obj.call_method1(method_name, (value, start)),
        (None, Some(stop)) => obj.call_method1(method_name, (value, stop)),
        (None, None) => obj.call_method1(method_name, (value,)),
    }
}
fn py_count<'py>(
    obj: &Bound<'py, PyAny>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    obj.call_method1(intern!(py, "count"), (value,))
}
