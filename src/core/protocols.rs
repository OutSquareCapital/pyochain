use pyo3::{
    PyClass, intern,
    prelude::*,
    types::{PyDict, PyFrozenSet, PyIterator, PyList, PyMapping, PyNone, PySet, PyTuple},
};
use pyo3_ext::{
    iter::CollectBoundIterator,
    prelude::{IntoPyIterator, PyDictExtConstructors, PyExtConstructors},
    types::PyIterable,
};
use pyochain_macros::{py_abc, try_cast_into};
use tap::Pipe;

use crate::{
    collections,
    core::{Dict, PyoVec, Seq, Set, SetMut, iterators},
    traits::{IntoInit, IntoPyochain, PyWrapper},
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
impl FromPyIter for Dict {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        into_dict(iterable)?.into_pyochain()
    }
}
impl FromPyIter for iterators::Iter {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        iterable.try_iter()?.into_pyochain()
    }
}
impl FromPyIter for Seq {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        try_cast_into! {
            match iterable {
                CaseExact::Self(slf) => Ok(slf),
                iterable => PyTuple::from_iterable(&iterable).and_then(Bound::into_pyochain),
            }
        }
    }
}
impl FromPyIter for PyoVec {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::Self(inner) => {
                    inner.get().inner_into_bound(py).as_sequence().to_list()?
                }
                iterable => PyList::from_iterable(&iterable)?,
            }
        }
        .into_pyochain()
    }
}
impl FromPyIter for Set {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::Self(inner) => inner
                    .get()
                    .inner_into_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                any => PyFrozenSet::from_iterable(&any)?,
            }
        }
        .into_pyochain()
    }
}
impl FromPyIter for SetMut {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::Self(inner) => inner
                    .get()
                    .inner_into_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                any => PySet::from_iterable(&any)?,
            }
        }
        .into_pyochain()
    }
}
impl FromPyIter for collections::StableSet {
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::PyDict(dict) => dict,
                CaseExact::Self(inner) => inner.get().inner_into_bound(py),
                iterable => PyDict::from_keys(iterable, None)?,
            }
        }
        .unbind()
        .pipe(Self)
        .into_bound(py)
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
            (Some(iterable), None) => into_dict(iterable)?,
            (Some(iterable), Some(kw)) => {
                let dict = into_dict(iterable)?;
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
        kwargs.unwrap_or_else(|| PyDict::new(py)).into_pyochain()
    }
}
impl FromPyArgs for iterators::Iter {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        match elements.len() {
            1 => try_cast_into! {match { unsafe { elements.get_item_unchecked(0) } } {
                Case::PyIterator(iterator) => iterator,
                Case::PyIterable(iterable) => iterable.try_iter()?,
                any => PyTuple::new(elements.py(), [any])?.iter_py(),
            }},
            _ => elements.iter_py(),
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.iter_py().into_pyochain()
    }
}
impl FromPyArgs for Seq {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        {
            match elements.len() {
                1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                    CaseExact::Self(inner) => inner.get().inner_into_bound(py),
                    Case::PyIterable(iterable) => PyTuple::from_iterable(&iterable)?,
                    any => PyTuple::new(py, [any])?,
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
        elements.into_pyochain()
    }
}
impl FromPyArgs for PyoVec {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyList::empty(py),
            1 => try_cast_into! {match unsafe {elements.get_item_unchecked(0)} {
                CaseExact::Self(inner) => {
                    inner.get().inner_into_bound(py).as_sequence().to_list()?
                }
                Case::PyIterable(iterable) => PyList::from_iterable(&iterable)?,
                any => PyList::new(py, [any])?,
            }},
            _ => elements.to_list(),
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.to_list().into_pyochain()
    }
}
impl FromPyArgs for collections::StableSet {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyDict::new(py),
            1 => {
                try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                Case::PyIterable(iterable) => PyDict::from_keys(iterable, None)?,
                any => {
                    let dict = PyDict::new(py);
                    dict.set_item(any, PyNone::get(py))?;
                    dict
                }}}
            }
            _ => PyDict::from_keys(elements.into_any(), None)?,
        }
        .unbind()
        .pipe(Self)
        .init()
        .pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_any()
            .pipe(|any| PyDict::from_keys(any, None))
            .map(Bound::unbind)
            .map(Self)
            .and_then(|slf| slf.into_bound(py))
    }
}
impl FromPyArgs for Set {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PyFrozenSet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                CaseExact::Self(inner) => inner
                    .get()
                    .inner_into_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                Case::PyIterable(iterable) => PyFrozenSet::from_iterable(&iterable)?,
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
            .into_pyochain()
    }
}
impl FromPyArgs for SetMut {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        match elements.len() {
            0 => PySet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                CaseExact::Self(inner) => inner
                    .get()
                    .inner_into_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                Case::PyIterable(iterable) => PySet::from_iterable(&iterable)?,
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
            .into_pyochain()
    }
}

#[inline]
fn into_dict<'py>(obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
    let py = obj.py();
    try_cast_into! {
        match obj {
            Case::PyDict(dict) => Ok(dict),
            Case::PyMapping(mapping) => PyDict::from_mapping(mapping),
            supports_keys
                if supports_keys.hasattr(intern!(py, "__getitem__"))?
                    && supports_keys.hasattr(intern!(py, "keys"))? =>
            {
                unsafe { supports_keys.cast_into_unchecked::<PyMapping>() }
                    .pipe(PyDict::from_mapping)
            }
            iterable => iterable.as_any().pipe(PyDict::from_sequence),
        }
    }
}
