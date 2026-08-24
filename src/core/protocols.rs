use pyo3::{
    PyClass,
    exceptions::PyTypeError,
    intern,
    prelude::*,
    types::{
        PyDict, PyFrozenSet, PyIterator, PyList, PyMapping, PyNone, PySequence, PySet, PyTuple,
    },
};
use pyo3_ext::{
    iter::{CollectBoundIterator, TryCollectBoundIterator},
    prelude::{IntoPyIterator, PyDictExtConstructors},
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

#[py_abc(Seq, PyoVec, Set, SetMut, collections::StableSet, iterators::Iter)]
pub trait FlexInitProtocol: Sized + PyClass {
    #[pyo3(signature = (*elements))]
    #[new]
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>>;
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>>;
    #[staticmethod]
    #[pyo3(signature = (*elements))]
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>>;
}
#[py_abc(Dict)]
pub trait FlexInitKwargsProtocol: Sized + PyClass {
    #[pyo3(signature = (iterable = None,/, **kwargs))]
    #[new]
    fn new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>>;
    #[pyo3(signature = (iterable, /))]
    #[staticmethod]
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>>;
    #[staticmethod]
    #[pyo3(signature = (**kwargs))]
    fn of<'py>(py: Python<'py>, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>>;
}
impl FlexInitKwargsProtocol for Dict {
    fn new(
        py: Python<'_>,
        iterable: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyClassInitializer<Self>> {
        let dict = match (iterable, kwargs) {
            (None, None) => PyDict::new(py),
            (None, Some(kw)) => kw,
            (Some(iterable), None) => into_dict(iterable)?,
            (Some(iterable), Some(kw)) => {
                let dict = into_dict(iterable)?;
                dict.update(kw.as_mapping())?;
                dict
            }
        };
        dict.unbind().pipe(Self).init().pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        into_dict(iterable)?.into_pyochain()
    }
    fn of<'py>(py: Python<'py>, kwargs: Option<Bound<'py, PyDict>>) -> PyResult<Bound<'py, Self>> {
        kwargs.unwrap_or_else(|| PyDict::new(py)).into_pyochain()
    }
}
impl FlexInitProtocol for iterators::Iter {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let iterator = match elements.len() {
            1 => try_cast_into! {match { unsafe { elements.get_item_unchecked(0) } } {
                Case::PyIterator(iterator) => iterator,
                Case::PyIterable(iterable) => iterable.try_iter()?,
                any => PyTuple::new(elements.py(), [any])?.iter_py(),
            }},
            _ => elements.iter_py(),
        };
        iterator.unbind().pipe(Self).init().pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.iter_py().into_pyochain()
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        iterable.try_iter()?.into_pyochain()
    }
}
impl FlexInitProtocol for Seq {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let tup = {
            match elements.len() {
                1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                    CaseExact::PyTuple(tuple) => tuple,
                    CaseExact::Self(inner) => inner.get().into_inner_bound(py),
                    Case::PySequence(sequence) => sequence.to_tuple()?,
                    Case::PyIterable(iterable) => iterable
                        .try_iter()?
                        .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?
                        .pipe(|x| PyTuple::new(py, x))?,
                    any => PyTuple::new(py, [any])?,
                }},
                _ => elements,
            }
        };
        tup.unbind().pipe(Self).init().pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::PyTuple(tuple) => tuple.into_pyochain(),
                CaseExact::Self(inner) => Ok(inner),
                Case::PySequence(sequence) => sequence.to_tuple()?.into_pyochain(),
                iterable => iterable
                    .try_iter()?
                    .collect::<PyResult<Vec<Bound<'_, PyAny>>>>()?
                    .pipe(|x| PyTuple::new(py, x))?
                    .into_pyochain(),
            }
        }
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.into_pyochain()
    }
}
impl FlexInitProtocol for PyoVec {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let list = match elements.len() {
            0 => PyList::empty(py),
            1 => try_cast_into! {match unsafe {elements.get_item_unchecked(0)} {
                CaseExact::Self(inner) => {
                    inner.get().into_inner_bound(py).as_sequence().to_list()?
                }
                Case::PySequence(sequence) => sequence.to_list()?,
                Case::PyIterable(iterable) => iterable.try_iter()?.try_collect_bound(py)?,
                any => PyList::new(py, [any])?,
            }},
            _ => elements.to_list(),
        };
        list.unbind().pipe(Self).init().pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        let list = try_cast_into! {
            match iterable {
                CaseExact::PyList(list) => list.as_sequence().to_list()?,
                CaseExact::Self(inner) => {
                    inner.get().into_inner_bound(py).as_sequence().to_list()?
                }
                Case::PySequence(sequence) => sequence.to_list()?,
                iterable => iterable.try_iter()?.try_collect_bound(py)?,
            }
        };
        list.into_pyochain()
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        elements.to_list().into_pyochain()
    }
}
impl FlexInitProtocol for collections::StableSet {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let dict = match elements.len() {
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
        };
        dict.unbind().pipe(Self).init().pipe(Ok)
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        let dict = try_cast_into! {
            match iterable {
                CaseExact::PyDict(dict) => dict,
                CaseExact::Self(inner) => inner.get().into_inner_bound(py),
                iterable => PyDict::from_keys(iterable, None)?,
            }
        };
        dict.unbind().pipe(Self).into_bound(py)
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
impl FlexInitProtocol for Set {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let set = match elements.len() {
            0 => PyFrozenSet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                CaseExact::PyFrozenSet(set) => set,
                CaseExact::Self(inner) => inner
                    .get()
                    .into_inner_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                Case::PyIterable(iterable) => iterable.try_iter()?.try_collect_bound(py)?,
                any => [any].into_iter().collect_bound(py)?,
            }},
            _ => elements.into_iter().collect_bound::<PyFrozenSet>(py)?,
        };
        set.unbind().pipe(Self).init().pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_iter()
            .collect_bound::<PyFrozenSet>(py)?
            .into_pyochain()
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::PyFrozenSet(set) => set,
                CaseExact::Self(inner) => inner
                    .get()
                    .into_inner_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                any => any.try_iter()?.try_collect_bound(py)?,
            }
        }
        .into_pyochain()
    }
}
impl FlexInitProtocol for SetMut {
    fn new(elements: Bound<'_, PyTuple>) -> PyResult<PyClassInitializer<Self>> {
        let py = elements.py();
        let set = match elements.len() {
            0 => PySet::empty(py)?,
            1 => try_cast_into! {match unsafe { elements.get_item_unchecked(0) } {
                CaseExact::PySet(set) => set.into_iter().collect_bound(py)?,
                CaseExact::Self(inner) => inner
                    .get()
                    .into_inner_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                Case::PyIterable(iterable) => iterable.try_iter()?.try_collect_bound(py)?,
                any => PySet::new(py, [any])?,
            }},
            _ => elements.into_iter().collect_bound::<PySet>(py)?,
        };
        set.unbind().pipe(Self).init().pipe(Ok)
    }
    fn of(elements: Bound<'_, PyTuple>) -> PyResult<Bound<'_, Self>> {
        let py = elements.py();
        elements
            .into_iter()
            .collect_bound::<PySet>(py)?
            .into_pyochain()
    }
    fn from_iter(iterable: Bound<'_, PyAny>) -> PyResult<Bound<'_, Self>> {
        let py = iterable.py();
        try_cast_into! {
            match iterable {
                CaseExact::PySet(set) => set.into_iter().collect_bound(py)?,
                CaseExact::Self(inner) => inner
                    .get()
                    .into_inner_bound(py)
                    .into_iter()
                    .collect_bound(py)?,
                any => any.try_iter()?.try_collect_bound(py)?,
            }
        }
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
            Case::PyIterable(iterable) => iterable.as_any().pipe(PyDict::from_sequence),
            incorrect_type => Err(PyTypeError::new_err(format!(
                "Cannot convert object of type {} to dict",
                incorrect_type.get_type().name()?
            ))),
        }
    }
}
