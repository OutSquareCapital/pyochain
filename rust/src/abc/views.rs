use pyo3::{
    PyTypeInfo,
    exceptions::PyKeyError,
    prelude::*,
    types::{PyIterator, PySet, PyType},
};
use tap::Pipe;

use crate::{
    abc::{PyoSet, PyoSized},
    iterators,
    mixins::Checkable,
    sets::SetMut,
    traits::{IntoPyochain, PyoABC},
};
#[pyclass(subclass, frozen, generic, extends=PyoSized)]
pub struct PyoMappingView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoMappingView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoSized)
            .add_subclass(Self {
                _mapping: mapping.unbind(),
            })
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoValuesView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoValuesView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init().add_subclass(Self {
            _mapping: mapping.unbind(),
        })
    }

    fn __contains__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let mapping = self._mapping.bind(value.py());
        mapping
            .try_iter()?
            .map(|key| mapping.get_item(&key?))
            .map(|item| item.and_then(|v| Ok(v.is(&value) || v.eq(&value)?)))
            .find_map(|item| match item {
                Ok(true) => Some(Ok(true)),
                Ok(false) => None,
                Err(err) => Some(Err(err)),
            })
            .unwrap_or_else(|| Ok(false))
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<iterators::ValuesViewIterator> {
        let py = slf.py();
        slf.get()
            ._mapping
            .clone_ref(py)
            .into_bound(py)
            .pipe(iterators::ValuesViewIterator::new)
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoKeysView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoKeysView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init().add_subclass(Self {
            _mapping: mapping.unbind(),
        })
    }

    #[classmethod]
    fn _from_iterable<'py>(
        cls: Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        PySet::type_object(cls.py())
            .call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self._mapping.bind(key.py()).contains(key)
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        slf.get()._mapping.bind(slf.py()).try_iter()
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoItemsView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoItemsView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init().add_subclass(Self {
            _mapping: mapping.unbind(),
        })
    }

    #[classmethod]
    fn _from_iterable<'py>(
        cls: Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        PySet::type_object(cls.py())
            .call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
            .and_then(Bound::into_pyochain)
    }

    fn __contains__(&self, item: (Bound<'_, PyAny>, Bound<'_, PyAny>)) -> PyResult<bool> {
        let (key, value) = item;
        let py = key.py();

        self._mapping
            .bind(py)
            .get_item(key)
            .and_then(|v| Ok(v.is(&value) || v.eq(&value)?))
            .or_else(|err| {
                if err.is_instance_of::<PyKeyError>(py) {
                    Ok(false)
                } else {
                    Err(err)
                }
            })
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<iterators::ItemsViewIterator> {
        let py = slf.py();
        slf.get()
            ._mapping
            .clone_ref(py)
            .into_bound(py)
            .pipe(iterators::ItemsViewIterator::new)
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other)
            .and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }
}
