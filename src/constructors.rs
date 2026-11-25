use std::cmp::Ordering;

use pyo3::{
    PyClass, PyTypeInfo,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyList, PyNone, PySet, PyTuple},
};
use pyo3_ext::{prelude::*, types::PyIterable};
use pyochain_macros::{py_abc, try_cast_into};
use tap::prelude::*;

use crate::{
    collections,
    core::{Dict, PyoVec, Seq, Set, SetMut, iterators},
};
#[pyclass(frozen, generic)]
pub struct FromIter;
#[pyclass(frozen, generic)]
pub struct ArgsWrapper;
#[pyclass(frozen, generic)]
pub struct KwargsWrapper;
#[pyclass(frozen, generic)]
pub struct Wrapper;
#[pyclass(frozen, generic)]
pub struct FromKwargs;
#[pyclass(frozen, generic)]
pub struct FromArgs;

#[py_abc(
    Seq,
    PyoVec,
    Set,
    SetMut,
    collections::StableSet,
    iterators::Iter,
    Dict
)]
pub trait FromPyIter: Sized {
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>>;
}
#[py_abc(Seq, PyoVec, Set, SetMut, collections::StableSet, iterators::Iter)]
pub trait FromPyArgs: PyClass + FromPyIter {
    #[pyo3(signature = (*elements))]
    #[new]
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>>;
    #[staticmethod]
    #[pyo3(signature = (*elements))]
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>>;
}
#[py_abc(Dict)]
pub trait FromPyKwargs: PyClass + FromPyIter {
    #[pyo3(signature = (iterable = None,/, **kwargs))]
    #[new]
    fn new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>>;
    #[staticmethod]
    #[pyo3(signature = (**kwargs))]
    fn of<'py>(py: Python<'py>, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>>;
}
macro_rules! impl_from_py_iter {
    ($($wrapped:ty => $T:ty),* $(,)?) => {
        $(
            impl FromPyIter for $T {

                #[inline(always)]
                fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, $T>> {
                    iterable.try_into_py::<$wrapped>().and_then(Bound::try_into_py)
                }
            }
        )*
    };
}
impl_from_py_iter!(PyDict => Dict, PyTuple => Seq, PyList => PyoVec, PyFrozenSet => Set, PySet => SetMut);
impl FromPyIter for iterators::Iter {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        iterable.try_iter().and_then(Bound::try_into_py)
    }
}
impl FromPyIter for collections::StableSet {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        PyDict::from_keys(&iterable)?
            .unbind()
            .conv::<Self>()
            .into_bound(iterable.py())
    }
}
impl FromPyKwargs for Dict {
    fn new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        match (iterable, kwargs) {
            (None, None) => PyDict::new(py),
            (None, Some(kw)) => kw,
            (Some(iterable), None) => iterable.try_into_py()?,
            (Some(iterable), Some(kw)) => {
                let dict = PyDict::try_from_py(iterable)?;
                dict.update(kw.as_mapping())?;
                dict
            }
        }
        .pipe(into_target)
        .pipe(Ok)
    }

    fn of<'py>(py: Python<'py>, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>> {
        kwargs.unwrap_or_else(|| PyDict::new(py)).try_into_py()
    }
}
impl FromPyArgs for iterators::Iter {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Value(elem) => tuple!(elem)?.iter_py(),
            Args::Same(slf) => slf.get().clone_ref(py).into_bound(py),
            Args::Iterable(it) => it.try_iter()?,
            Args::Empty(tuple) | Args::Multiple(tuple) => tuple.iter_py(),
        }
        .pipe(into_target)
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.iter_py().try_into_py()
    }
}
impl FromPyArgs for Seq {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Value(elem) => tuple!(elem),
            Args::Same(slf) => slf.get().clone_ref(py).into_bound(py).pipe(Ok),
            Args::Iterable(it) => it.try_into_py(),
            Args::Empty(tuple) | Args::Multiple(tuple) => Ok(tuple),
        }
        .map(into_target)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.try_into_py()
    }
}
impl FromPyArgs for PyoVec {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Empty(_) => Ok(PyList::empty(py)),
            Args::Same(slf) => slf.get().bind(py).copy(),
            Args::Iterable(it) => it.try_into_py(),
            Args::Value(elem) => list![elem],
            Args::Multiple(tuple) => tuple.to_list().pipe(Ok),
        }
        .map(into_target)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.to_list().try_into_py()
    }
}
impl FromPyArgs for collections::StableSet {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Empty(_) => Ok(PyDict::new(py)),
            Args::Same(slf) => slf.get().bind(py).copy(),
            Args::Iterable(iterable) => PyDict::from_keys(&iterable),
            Args::Value(elem) => {
                let dict = PyDict::new(py);
                dict.set_item(elem, PyNone::get(py))?;
                Ok(dict)
            }
            Args::Multiple(tuple) => PyDict::from_keys(&tuple),
        }
        .map(into_target)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements
            .pipe_ref(PyDict::from_keys)?
            .unbind()
            .conv::<Self>()
            .into_bound(elements.py())
    }
}
impl FromPyArgs for Set {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Empty(_) => PyFrozenSet::empty(py),
            Args::Same(inner) => inner.get().clone_ref(py).into_bound(py).pipe(Ok),
            Args::Iterable(iterable) => iterable.try_into_py(),
            Args::Value(elem) => [elem].into_iter().collect_bound(py),
            Args::Multiple(tuple) => tuple.into_iter().collect_bound(py),
        }
        .map(into_target)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_iter()
            .collect_bound::<PyFrozenSet>(py)?
            .try_into_py()
    }
}
impl FromPyArgs for SetMut {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match Args::<Self>::new(elements) {
            Args::Empty(_) => PySet::empty(py),
            Args::Same(inner) => inner.get().bind(py).copy(),
            Args::Iterable(it) => it.try_into_py(),
            Args::Value(elem) => PySet::new(py, [elem]),
            Args::Multiple(tuple) => tuple.into_iter().collect_bound(py),
        }
        .map(into_target)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_iter()
            .collect_bound::<PySet>(py)?
            .try_into_py()
    }
}
#[inline(always)]
fn into_target<T: Init + From<Py<O>>, O>(obj: Bound<'_, O>) -> PyClassInitializer<T> {
    obj.unbind().conv::<T>().init()
}
#[derive(Debug)]
enum Args<'py, T> {
    Empty(Bound<'py, PyTuple>),
    Same(Bound<'py, T>),
    Iterable(Bound<'py, PyIterable>),
    Value(Bound<'py, PyAny>),
    Multiple(Bound<'py, PyTuple>),
}
impl<'py, T: PyTypeInfo> Args<'py, T> {
    #[inline]
    fn new(tuple: Bound<'py, PyTuple>) -> Self {
        match tuple.len().cmp(&1) {
            Ordering::Less => Self::Empty(tuple),
            Ordering::Equal => try_cast_into! {match unsafe { tuple.get_item_unchecked(0) } {
                CaseExact::T(inner) => Self::Same(inner),
                Case::PyIterable(iterable) => Self::Iterable(iterable),
                any => Self::Value(any),
            }},
            Ordering::Greater => Self::Multiple(tuple),
        }
    }
}
