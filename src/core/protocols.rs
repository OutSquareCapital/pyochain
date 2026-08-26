use pyo3::{
    PyClass,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyIterator, PyList, PyNone, PySet, PyTuple},
};
use pyo3_ext::{prelude::*, types::PyIterable};
use pyochain_macros::{py_abc, try_cast_into};
use tap::Pipe;

use crate::{
    collections,
    core::{Dict, PyoVec, Seq, Set, SetMut, iterators},
    traits::{IntoInit, PyWrapper},
};
#[pyclass(module = "pyochain.core", frozen, generic)]
pub struct FlexibleInit;
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
pub trait FromPyArgs: Sized + PyClass + FromPyIter {
    #[pyo3(signature = (*elements))]
    #[new]
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>>;
    #[staticmethod]
    #[pyo3(signature = (*elements))]
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>>;
}
#[py_abc(Dict)]
pub trait FromPyKwargs: Sized + PyClass + FromPyIter {
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
            .pipe(Self)
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
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }

    fn of<'py>(py: Python<'py>, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>> {
        kwargs.unwrap_or_else(|| PyDict::new(py)).try_into_py()
    }
}
impl FromPyArgs for iterators::Iter {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        match elements.len() {
            1 => try_cast_into! {match { unsafe { elements.get_item_unchecked(0) } } {
                Case::PyIterator(iterator) => iterator,
                Case::PyIterable(iterable) => iterable.try_iter()?,
                any => tuple!(any)?.iter_py(),
            }},
            _ => elements.iter_py(),
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.iter_py().try_into_py()
    }
}
impl FromPyArgs for Seq {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        {
            match elements.len() {
                1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                    CaseExact::Self(inner) => inner.get().inner_into_bound(py),
                    Case::PyIterable(iterable) => iterable.try_into_py::<PyTuple>()?,
                    any => tuple!(any)?,
                }},
                _ => elements,
            }
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.try_into_py()
    }
}
impl FromPyArgs for PyoVec {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyList::empty(py),
            1 => try_cast_into! {match unsafe {elements.get_item_unchecked(0)} {
                Case::PyIterable(iterable) => iterable.try_into_py::<PyList>()?,
                any => list![any]?,
            }},
            _ => elements.to_list(),
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.to_list().try_into_py()
    }
}
impl FromPyArgs for collections::StableSet {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyDict::new(py),
            1 => {
                try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                Case::PyIterable(iterable) => PyDict::from_keys(&iterable)?,
                any => {
                    let dict = PyDict::new(py);
                    dict.set_item(any, PyNone::get(py))?;
                    dict
                }}}
            }
            _ => PyDict::from_keys(elements.as_any())?,
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements
            .as_any()
            .pipe(PyDict::from_keys)
            .map(Bound::unbind)
            .map(Self)
            .and_then(|slf| slf.into_bound(elements.py()))
    }
}
impl FromPyArgs for Set {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyFrozenSet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                CaseExact::Self(inner) => inner.get().inner_into_bound(py),
                Case::PyIterable(iterable) => iterable.try_into_py::<PyFrozenSet>()?,
                any => [any].into_iter().collect_bound(py)?,
            }},
            _ => elements.into_iter().collect_bound::<PyFrozenSet>(py)?,
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
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
        match elements.len() {
            0 => PySet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                Case::PyIterable(iterable) => iterable.try_into_py::<PySet>()?,
                any => PySet::new(py, [any])?,
            }},
            _ => elements.into_iter().collect_bound::<PySet>(py)?,
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_iter()
            .collect_bound::<PySet>(py)?
            .try_into_py()
    }
}
