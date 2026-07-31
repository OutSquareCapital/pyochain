use either::Either;
use pyo3::{
    BoundObject, IntoPyObjectExt, PyTypeInfo,
    exceptions::PyKeyError,
    intern,
    prelude::*,
    types::{PyDict, PyIterator, PyMapping, PyNone, PyNotImplemented, PySet, PyTuple, PyType},
};
use tap::Pipe;

use crate::{
    abc::{PyoCollection, PyoSet, PyoSized},
    mixins::Checkable,
    option::{PyNull, PySome},
    pyo3_ext::{
        args::{Args, Kwargs},
        types::PyCmpOut,
    },
    result::{PyoErr, PyoOk},
    sets::SetMut,
    iterators,
    traits::{IntoPyochain, PyoABC},
};
#[pyclass(subclass, frozen, generic, mapping, extends=PyoCollection)]
pub struct PyoMapping;
#[pymethods]
impl PyoMapping {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn __contains__(slf: Bound<'_, Self>, key: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.get_item(key).map(|_| true).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(slf.py()) {
                Ok(false)
            } else {
                Err(err)
            }
        })
    }

    fn __eq__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyCmpOut<bool, 'py> {
        let py = slf.py();
        match other.cast::<PyMapping>() {
            Ok(other) => slf
                .call_method0(intern!(py, "items"))?
                .pipe_ref(PyDict::from_sequence)?
                .eq(other
                    .call_method0(intern!(py, "items"))?
                    .pipe_ref(PyDict::from_sequence)?)
                .map(Either::Left),
            Err(_) => PyNotImplemented::get(py)
                .into_bound()
                .pipe(Ok)
                .map(Either::Right),
        }
    }

    fn keys(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoKeysView>> {
        PyoKeysView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoKeysView>() })
    }

    fn values(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoValuesView>> {
        PyoValuesView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoValuesView>() })
    }

    fn items(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoItemsView>> {
        PyoItemsView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoItemsView>() })
    }

    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(py) {
                Ok(default.unwrap_or_else(|| PyNone::get(py).into_bound_py_any(py).unwrap()))
            } else {
                Err(err)
            }
        })
    }

    fn get_item<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key)
            .and_then(|value| PySome::new(value.unbind()).into_bound_py_any(py))
            .or_else(|err| {
                if err.is_instance_of::<PyKeyError>(py) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            })
    }
}

#[pyclass(subclass, frozen, generic, mapping, extends=PyoMapping)]
pub struct PyoMutableMapping;

#[pymethods]
impl PyoMutableMapping {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match slf.get_item(key) {
            Ok(value) => {
                slf.del_item(key)?;
                Ok(value)
            }
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    default.ok_or(err)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn popitem(slf: Bound<'_, Self>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
        slf.try_iter()?
            .next()
            .map(|k| {
                let key = k?;
                let value = slf.get_item(&key)?;
                slf.del_item(&key)?;
                Ok((key, value))
            })
            .unwrap_or_else(|| Err(PyKeyError::new_err("")))
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        loop {
            match slf.call_method0(intern!(py, "popitem")) {
                Ok(_) => continue,
                Err(err) => {
                    if err.is_instance_of::<PyKeyError>(py) {
                        return Ok(());
                    } else {
                        return Err(err);
                    }
                }
            }
        }
    }

    #[pyo3(signature = (other=None, **kwds))]
    fn update(
        slf: Bound<'_, Self>,
        other: Option<Bound<'_, PyAny>>,
        kwds: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        other.map(|x| {
            if x.is_instance_of::<PyMapping>() {
                x.try_iter()?
                    .try_for_each(|key| key.and_then(|k| slf.set_item(&k, x.get_item(&k)?)))
            } else if x.hasattr("keys")? {
                x.call_method0(intern!(slf.py(), "keys"))
                    .unwrap()
                    .try_iter()?
                    .try_for_each(|key| key.and_then(|k| slf.set_item(&k, x.get_item(&k)?)))
            } else {
                x.try_iter()?.try_for_each(|item| {
                    let tup = item?.cast_into::<PyTuple>()?;
                    let (key, value) = (tup.get_item(0)?, tup.get_item(1)?);
                    slf.set_item(&key, &value)
                })
            }
        });
        kwds.map(|kwds| {
            kwds.items()
                .iter()
                .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
                .map(|x| unsafe { (x.get_item_unchecked(0), x.get_item_unchecked(1)) })
                .try_for_each(|(key, value)| slf.set_item(&key, &value))
        });
        Ok(())
    }
    #[pyo3(signature = (key, default=None))]
    fn setdefault<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(py) {
                let default = default
                    .map(Ok)
                    .unwrap_or_else(|| PyNone::get(py).into_bound_py_any(py))?;
                slf.set_item(key, &default)?;
                Ok(default)
            } else {
                Err(err)
            }
        })
    }

    fn insert<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let previous = slf.get_item(key);
        slf.set_item(&key, &value)?;
        previous
            .map(|x| PySome::new(x.unbind()).into_bound_py_any(py))
            .unwrap_or_else(|_| PyNull::get(py).into_bound_py_any(py))
    }

    fn try_insert<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if slf.contains(&key)? {
            PyoErr::new(
                PyKeyError::new_err(format!(
                    "Key {} already exists with value {}.",
                    key,
                    slf.get_item(&key)?
                ))
                .into_py_any(py)?,
            )
            .into_bound_py_any(py)
        } else {
            slf.set_item(&key, &value)?;
            value.unbind().pipe(PyoOk::new).into_bound_py_any(py)
        }
    }

    fn remove<'py>(slf: Bound<'py, Self>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key)
            .and_then(|value| {
                slf.del_item(key)?;
                value.unbind().pipe(PySome::new).into_bound_py_any(py)
            })
            .or_else(|err| {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            })
    }

    fn remove_entry<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        match slf.get_item(key) {
            Ok(value) => {
                slf.del_item(key)?;
                PyTuple::new(py, [key, &value])?
                    .into_any()
                    .unbind()
                    .pipe(PySome::new)
                    .into_bound_py_any(py)
            }
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            }
        }
    }
}

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
        for item in mapping.try_iter()?.map(|key| mapping.get_item(&key?)) {
            let v = item?;
            if v.is(&value) || v.eq(&value)? {
                return Ok(true);
            }
        }
        Ok(false)
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
    ) -> PyResult<Bound<'py, PySet>> {
        PySet::type_object(cls.py())
            .call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
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
    ) -> PyResult<Bound<'py, PySet>> {
        PySet::type_object(cls.py())
            .call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
    }

    fn __contains__(&self, item: (Bound<'_, PyAny>, Bound<'_, PyAny>)) -> PyResult<bool> {
        let (key, value) = item;
        let py = key.py();

        let v = self
            ._mapping
            .bind(py)
            .get_item(key)
            .and_then(|v| Ok(v.is(&value) || v.eq(&value)?));
        match v {
            Ok(v) => Ok(v),
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(py) {
                    Ok(false)
                } else {
                    Err(err)
                }
            }
        }
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
