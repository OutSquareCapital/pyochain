use crate::{abc, mixins, pylibs};
use either::Either;
use pyo3::pyclass_init::PyClassInitializer;
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyDict, PyInt, PyIterator, PyList, PyRange, PyRangeMethods, PySequence, PyString, PyTuple,
};
use pyo3::{PyTypeInfo, ffi, intern, prelude::*};
use tap::Pipe;
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
        data.pipe(|x| PyTuple::type_object(py).call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
            .map(Bound::unbind)
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
    fn __lt__(&self, value: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.inner.bind(value.py()).lt(value)
    }
    fn __le__(&self, value: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.inner.bind(value.py()).le(value)
    }
    fn __gt__(&self, value: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.inner.bind(value.py()).gt(value)
    }
    fn __ge__(&self, value: &Bound<'_, PyTuple>) -> PyResult<bool> {
        self.inner.bind(value.py()).ge(value)
    }
    fn __add__<'py>(&self, value: &Bound<'py, PyTuple>) -> PyResult<Bound<'py, PyTuple>> {
        self.inner
            .bind(value.py())
            .as_sequence()
            .concat(value.as_sequence())
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
    }
    fn __mul__(slf: Bound<'_, Self>, value: usize) -> PyResult<Bound<'_, PyTuple>> {
        slf.get()
            .inner
            .bind(slf.py())
            .as_sequence()
            .repeat(value)
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
    }
    fn __rmul__<'py>(
        slf: Bound<'py, Self>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let py = value.py();
        value
            .mul(slf.get().inner.bind(py).as_any())
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
    }
    #[pyo3(signature = (value, /))]
    fn count(&self, value: Bound<'_, PyAny>) -> PyResult<usize> {
        self.inner.bind(value.py()).as_sequence().count(value)
    }
    #[pyo3(signature = (value, start = None, stop = None, /))]
    fn index<'py>(
        &self,
        value: &Bound<'py, PyAny>,
        start: Option<usize>,
        stop: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = value.py();
        self.inner.bind(py).call_method1(
            intern!(py, "index"),
            (value, start.unwrap_or(0), stop.unwrap_or(usize::MAX)),
        )
    }

    fn repeat(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        slf.get()
            .inner
            .bind(py)
            .as_sequence()
            .repeat(n)
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn concat<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let cls = slf.get_type();
        let other_seq = other
            .cast_exact::<PyTuple>()
            .map(|x| x.as_unbound().clone_ref(py))
            .or_else(|_| {
                other
                    .cast_exact::<Self>()
                    .map(|x| x.get().inner.clone_ref(py))
            })
            .map_err(PyErr::from)?
            .into_bound(py)
            .into_sequence();
        slf.get()
            .inner
            .bind(py)
            .as_sequence()
            .concat(&other_seq)
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
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
        let py = data.py();
        let initializer = PyClassInitializer::from(mixins::Checkable)
            .add_subclass(abc::PyoIterable)
            .add_subclass(abc::PyoCollection)
            .add_subclass(abc::PyoSequence)
            .add_subclass(abc::PyoMutableSequence)
            .add_subclass(Self {
                inner: data.unbind(),
            });
        Bound::new(py, initializer)
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
        let o = extract_union::<Self, PyList>(&other)?
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
        extract_union::<Self, PyList>(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .map(|x| {
                slf.get()
                    .inner
                    .bind(py)
                    .as_sequence()
                    .concat(x.as_sequence())
            })
            .into_inner()
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Self::from_ref)
    }

    fn __iadd__<'py>(slf: Bound<'py, Self>, value: &Bound<'py, PySequence>) -> PyResult<()> {
        slf.get()
            .inner
            .bind(value.py())
            .as_sequence()
            .in_place_concat(value)?;
        Ok(())
    }

    fn __mul__(slf: Bound<'_, Self>, value: usize) -> PyResult<Bound<'_, Self>> {
        Self::from_ref(
            slf.get()
                .inner
                .bind(slf.py())
                .as_sequence()
                .repeat(value)
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
        )
    }

    fn __rmul__<'py>(
        slf: Bound<'py, Self>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::from_ref(
            value
                .mul(slf.get().inner.bind(value.py()))
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
        )
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
        let other = extract_union::<Self, PyList>(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).gt(other)
    }

    fn __ge__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = extract_union::<Self, PyList>(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).ge(other)
    }

    fn __lt__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = extract_union::<Self, PyList>(&value)?
            .map_left(|vec| vec.get().inner.bind(py))
            .into_inner();
        self.inner.bind(py).lt(other)
    }

    fn __le__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = value.py();
        let other = extract_union::<Self, PyList>(&value)?
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

    fn repeat(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        Self::from_ref(
            slf.get()
                .inner
                .bind(slf.py())
                .as_sequence()
                .repeat(n)
                .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })?,
        )
    }

    fn repeat_mut(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner
            .bind(slf.py())
            .as_sequence()
            .in_place_repeat(n)
            .map(|_| slf)
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

    fn concat<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        extract_union::<Self, PyList>(&other)?
            .map_left(|o| o.get().inner.bind(py))
            .into_inner()
            .pipe(|other| {
                slf.get()
                    .inner
                    .bind(py)
                    .as_sequence()
                    .concat(other.as_sequence())
            })
            .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .and_then(Self::from_ref)
    }

    fn concat_mut<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let other = extract_union::<Self, PyList>(&other)?
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
/// Extract the type of a value and check if it is one of two types, returning an `Either` with the result.\
/// Returns a `PyErr` if the value is not one of the two types.
#[inline]
fn extract_union<'py, 'r, T, U>(
    value: &'r Bound<'py, PyAny>,
) -> PyResult<Either<&'r Bound<'py, T>, &'r Bound<'py, U>>>
where
    T: PyTypeInfo,
    U: PyTypeInfo,
{
    value
        .cast_exact::<T>()
        .map(Either::Left)
        .or_else(|_| value.cast_exact::<U>().map(Either::Right))
        .map_err(PyErr::from)
}
