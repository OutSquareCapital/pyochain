use pyo3::{
    PyTypeInfo,
    exceptions::PyKeyError,
    prelude::*,
    types::{PyIterator, PySet, PyType},
};
use tap::Pipe;

use crate::{
    abc::{Checkable, PyoSet, PyoSized, traits::MappingView},
    core::{SetMut, iterators},
    traits::{IntoInit, IntoPyochain},
};
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSized)]
pub struct PyoMappingView(pub Py<PyAny>);
#[pymethods]
impl PyoMappingView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoSized)
            .add_subclass(Self(mapping.unbind()))
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSet)]
pub struct PyoValuesView(pub Py<PyAny>);
#[pymethods]
impl PyoValuesView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        Self(mapping.unbind()).init()
    }

    fn __contains__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let mapping = self.mapping().bind(value.py());
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
            .mapping()
            .clone_ref(py)
            .into_bound(py)
            .pipe(iterators::ValuesViewIterator::new)
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSet)]
pub struct PyoKeysView(pub Py<PyAny>);
#[pymethods]
impl PyoKeysView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        Self(mapping.unbind()).init()
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
        self.mapping().bind(key.py()).contains(key)
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        slf.get().mapping().bind(slf.py()).try_iter()
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSet)]
pub struct PyoItemsView(pub Py<PyAny>);
#[pymethods]
impl PyoItemsView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        Self(mapping.unbind()).init()
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
    /// NOTE: There's a fundamental incoherence between Python dict_items and `collections.abc.ItemsView` regarding the `__contains__` method.\
    /// dict_items will return `False` if the argument is not a tuple of length 2,\
    /// while `ItemsView` will try to unpack `item` as an `Iterable` of 2 elements.\
    /// Thus, the latter will work on ANY Iterable for truthyness, BUT will raise a `ValueError` on length issues,\
    /// while the former will return `False` on ANY non-tuple argument, and will NOT raise anything regarding length issues.\
    /// At the same time, the `typeshed` signature clearly stipulate that a `ItemsView` expect a `tuple` of 2 elements, and since `dict_items` is a virtual subclass of `ItemsView`, it should follow the same signature.\
    /// Thus, I choose to follow dict_items behavior.
    fn __contains__(&self, item: Bound<'_, PyAny>) -> PyResult<bool> {
        match item.extract::<(Bound<'_, PyAny>, Bound<'_, PyAny>)>() {
            Ok((key, value)) => {
                let py = key.py();
                self.mapping()
                    .bind(py)
                    .get_item(&key)
                    .and_then(|v| Ok(v.is(&value) || v.eq(&value)?))
                    .or_else(|err| {
                        if err.is_instance_of::<PyKeyError>(py) {
                            Ok(false)
                        } else {
                            Err(err)
                        }
                    })
            }
            Err(_) => Ok(false),
        }
    }
    fn __iter__(slf: Bound<'_, Self>) -> PyResult<iterators::ItemsViewIterator> {
        let py = slf.py();
        slf.get()
            .mapping()
            .clone_ref(py)
            .into_bound(py)
            .pipe(iterators::ItemsViewIterator::new)
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<SetMut>() })
    }
}
