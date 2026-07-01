use crate::{abc, mixins, pylibs};
use pyo3::sync::PyOnceLock;
use pyo3::types::{
    PyDict, PyInt, PyIterator, PyList, PyRange, PyRangeMethods, PySequence, PyString, PyTuple,
};
use pyo3::{PyTypeInfo, intern, prelude::*};
use tap::Pipe;
#[pyclass(frozen, generic, extends=abc::PyoSequence)]
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
            .pipe(get_repr)?;
        format!("{}({})", name, repr).pipe(Ok)
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> Bound<'py, PyIterator> {
        self.inner.clone_ref(py).bind(py).try_iter().unwrap()
    }

    fn __len__(&self, py: Python) -> usize {
        self.inner.clone_ref(py).bind(py).len()
    }

    fn __getitem__<'py>(&self, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        self.inner
            .clone_ref(py)
            .into_bound(py)
            .into_any()
            .get_item(index)
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> bool {
        let py = other.py();
        let left = self.inner.clone_ref(py).into_bound(py);
        if let Ok(o) = other.cast_exact::<Self>() {
            left.eq(o.get().inner.clone_ref(py).bind(py)).unwrap()
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

    fn __reversed__(slf: Bound<'_, Self>) -> Bound<'_, PyAny> {
        let py = slf.py();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }

    fn repeat(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        slf.get()
            .inner
            .clone_ref(py)
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
            .clone_ref(py)
            .into_bound(py)
            .into_sequence()
            .concat(&other_seq)
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}

#[pyclass(frozen, extends=abc::PyoSequence)]
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
            .clone_ref(py)
            .bind(py)
            .pipe(|x| unsafe { x.cast_unchecked::<PySequence>() })
            .len()
            .unwrap()
    }

    fn __getitem__<'py>(&self, index: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        self.inner
            .clone_ref(py)
            .into_bound(py)
            .into_any()
            .get_item(index)
    }

    fn __repr__(slf: Bound<'_, Self>) -> String {
        let py = slf.py();
        let name = slf.get_type().name().unwrap();
        let inner = slf.get().inner.clone_ref(py).into_bound(py);

        let params = format!(
            "{}, {}, {}",
            inner.start().unwrap(),
            inner.stop().unwrap(),
            inner.step().unwrap()
        );
        format!("{}({})", name, params)
    }

    fn __reversed__(slf: Bound<'_, Self>) -> Bound<'_, PyAny> {
        let py = slf.py();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }
}

#[pyclass(sequence, frozen, generic, name="Vec",extends=abc::PyoMutableSequence)]
pub struct PyoVec {
    #[pyo3(get)]
    inner: Py<PyList>,
}
#[pymethods]
impl PyoVec {
    #[new]
    fn new(data: Bound<'_, PyAny>) -> PyResult<PyClassInitializer<Self>> {
        let py = data.py();
        PyList::type_object(py)
            .call1((&data,))?
            .pipe(|x| unsafe { x.cast_into_unchecked::<PyList>() })
            .unbind()
            .pipe(|inner| {
                PyClassInitializer::from(mixins::Checkable)
                    .add_subclass(abc::PyoIterable)
                    .add_subclass(abc::PyoCollection)
                    .add_subclass(abc::PyoSequence)
                    .add_subclass(abc::PyoMutableSequence)
                    .add_subclass(Self { inner })
            })
            .pipe(Ok)
    }
    #[staticmethod]
    fn from_ref<'py>(py: Python<'py>, data: Bound<'py, PyList>) -> PyResult<Py<Self>> {
        data.unbind()
            .pipe(|inner| {
                PyClassInitializer::from(mixins::Checkable)
                    .add_subclass(abc::PyoIterable)
                    .add_subclass(abc::PyoCollection)
                    .add_subclass(abc::PyoSequence)
                    .add_subclass(abc::PyoMutableSequence)
                    .add_subclass(Self { inner })
            })
            .pipe(|init| Py::new(py, init))
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
            .pipe(get_repr)?;
        format!("{}({})", name, repr).pipe(Ok)
    }

    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        index: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = index.py();
        slf.get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .into_any()
            .get_item(index)
    }

    fn __setitem__<'py>(
        slf: &Bound<'py, Self>,
        index: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let py = index.py();
        slf.get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .into_any()
            .set_item(index, value)
    }

    fn __delitem__<'py>(slf: &Bound<'py, Self>, index: &Bound<'py, PyAny>) -> PyResult<()> {
        let py = index.py();
        slf.get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .into_any()
            .del_item(index)
    }

    fn __len__(slf: &Bound<'_, Self>) -> usize {
        let py = slf.py();
        slf.get().inner.clone_ref(py).into_bound(py).len()
    }

    fn __eq__(slf: &Bound<'_, Self>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let left = slf.get().inner.clone_ref(slf.py()).into_bound(slf.py());
        if let Ok(other) = other.cast_exact::<Self>() {
            left.eq(&other.get().inner)
        } else if let Ok(other) = other.cast_exact::<PyList>() {
            left.eq(&other)
        } else {
            Ok(false)
        }
    }
    fn __reversed__(slf: Bound<'_, Self>) -> Bound<'_, PyAny> {
        let py = slf.py();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .pipe_as_ref(pylibs::builtins::reversed)
    }
    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .call_method0(intern!(py, "copy"))
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn repeat(slf: Bound<'_, Self>, n: usize) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        slf.get()
            .inner
            .clone_ref(py)
            .bind(py)
            .as_sequence()
            .repeat(n)
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn repeat_mut(slf: Bound<'_, Self>, n: usize) -> PyResult<()> {
        let py = slf.py();
        slf.get().inner.bind(py).as_sequence().in_place_repeat(n)?;
        Ok(())
    }

    fn insert(&self, index: usize, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = value.py();
        self.inner.bind(py).insert(index, value)
    }
    fn concat<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let right = match other.cast_exact::<Self>() {
            Ok(vec) => vec.get().inner.bind(py),
            Err(_) => other.cast_exact::<PyList>()?,
        };
        slf.get()
            .inner
            .clone_ref(py)
            .into_bound(py)
            .as_sequence()
            .concat(right.as_sequence())
            .and_then(|data| slf.get_type().call1((data,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn concat_mut<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let right = if let Ok(other) = other.cast_exact::<Self>() {
            other.get().inner.clone_ref(py).into_bound(py)
        } else {
            other.cast_into_exact::<PyList>()?
        };
        slf.get()
            .inner
            .bind(py)
            .as_sequence()
            .in_place_concat(right.as_sequence())
            .map(|_| slf)
    }

    #[pyo3(signature = (*, reverse=false))]
    fn sort(slf: Bound<'_, Self>, reverse: bool) -> PyResult<Bound<'_, Self>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("reverse", reverse)?;
        slf.get()
            .inner
            .bind(py)
            .call_method(intern!(py, "sort"), (), Some(&kwargs))
            .map(|_| slf)
    }
    #[pyo3(signature = (key,*, reverse=false))]
    fn sort_by<'py>(
        slf: Bound<'py, Self>,
        key: Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let kwargs = PyDict::new(py);
        kwargs.set_item("key", key)?;
        kwargs.set_item("reverse", reverse)?;
        slf.get()
            .inner
            .bind(py)
            .call_method(intern!(py, "sort"), (), Some(&kwargs))
            .map(|_| slf)
    }
}
fn get_repr(obj: Bound<'_, PySequence>) -> PyResult<Bound<'_, PyString>> {
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
