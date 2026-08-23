use crate::{
    abc::{self, traits::ImplPyoReversible},
    core::iterators,
    display::pformat,
    traits::{IntoPyochain, PyWrapper},
};
use pyo3::{
    exceptions::PyKeyError,
    intern,
    prelude::*,
    types::{PyDict, PyIterator, PyTuple, PyType},
};
use pyo3_ext::{prelude::*, pylibs, types::PopResult};
use tap::Pipe;

#[pyclass(module = "pyochain.core",frozen, generic, extends=abc::PyoMutableMapping)]
pub struct Dict(pub Py<PyDict>);
#[pymethods]
impl Dict {
    #[allow(unused_variables)]
    #[classmethod]
    #[pyo3(signature = (keys, value=None))]
    fn from_keys<'py>(
        cls: Bound<'py, PyType>,
        keys: Bound<'py, PyAny>,
        value: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        PyDict::from_keys(keys, value)?.into_pyochain()
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let name = slf.get_type().name()?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("sort_dicts", false)?;
        let repr = pformat(py, slf.get().inner().clone_ref(py), false).map(|x| {
            let rs_str = x.to_string();
            let length = rs_str.len();
            rs_str[1..length - 1].to_string()
        })?;

        Ok(format!("{}({})", name, repr))
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, PyIterator> {
        slf.get().inner_bind(slf.py()).iter_py()
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self.inner_bind(key.py()).contains(key)
    }

    fn __len__(slf: Bound<'_, Self>) -> usize {
        slf.get().inner_bind(slf.py()).len()
    }

    fn __getitem__<'py>(&self, key: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        self.inner_bind(key.py()).as_any().get_item(key)
    }

    fn __setitem__(&self, key: Bound<'_, PyAny>, value: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(key.py()).set_item(key, value)
    }

    fn __delitem__(&self, key: Bound<'_, PyAny>) -> PyResult<()> {
        self.inner_bind(key.py()).del_item(key)
    }

    fn __eq__(&self, other: Bound<'_, PyAny>) -> bool {
        let py = other.py();
        Self::extract_union(&other)
            .and_then(|r| self.inner_bind(py).eq(r))
            .unwrap_or(false)
    }

    fn __or__<'py>(slf: Bound<'py, Self>, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        Self::union(slf, value)
    }

    fn __ror__<'py>(&self, value: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = value.py();
        Self::extract_union(&value)
            .and_then(|r| r.bitor(self.inner_bind(py)))
            .and_then(|new| unsafe { new.cast_into_unchecked::<PyDict>() }.into_pyochain())
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, value: Bound<'py, PyAny>) -> PyResult<()> {
        Self::union_mut(slf, value)?;
        Ok(())
    }

    #[staticmethod]
    fn from_object<'py>(obj: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        obj.getattr(intern!(obj.py(), "__dict__"))
            .and_then(|x| unsafe { x.cast_into_unchecked::<PyDict>() }.into_pyochain())
    }

    fn copy(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        slf.get()
            .inner_bind(slf.py())
            .copy()
            .and_then(Bound::into_pyochain)
    }

    #[pyo3(signature = (key, default=None, /))]
    fn pop<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        match self.inner_bind(py).pop_or_err(&key) {
            PopResult::Ok(v) => Ok(v),
            PopResult::Err(e) => Err(e),
            PopResult::KeyMissing => match default {
                Some(d) => Ok(d),
                None => Err(PyKeyError::new_err(key.to_string())),
            },
        }
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let rhs = Self::extract_union(&other)?;
        slf.get()
            .inner_bind(py)
            .bitor(rhs)
            .and_then(|new| unsafe { new.cast_into_unchecked::<PyDict>() }.into_pyochain())
    }

    fn union_mut<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = other.py();
        let lhs = slf.get().inner_bind(py);
        other
            .cast_exact::<Self>()
            .map(|x| lhs.ior(x.get().inner_bind(py).as_any()))
            .unwrap_or_else(|_| lhs.ior(&other))
            .map(|_| slf)
    }

    fn popitem(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyTuple>> {
        let py = slf.py();
        slf.get()
            .inner_bind(py)
            .call_method0(intern!(py, "popitem"))
            .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
    }

    fn clear(slf: Bound<'_, Self>) -> () {
        slf.get().inner_bind(slf.py()).clear()
    }
    #[pyo3(signature = (m=None, /, **kwargs))]
    fn update(
        &self,
        m: Option<Bound<'_, PyAny>>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        match (m, kwargs) {
            (None, None) => Ok(()),
            (None, Some(kwargs)) => self.inner_bind(kwargs.py()).update(kwargs.as_mapping()),
            (Some(m), _) => self
                .inner_bind(m.py())
                .call_method(intern!(m.py(), "update"), (m,), kwargs)
                .map(|_| ()),
        }
    }
    #[pyo3(signature = (key, default=None, /))]
    fn setdefault<'py>(
        &self,
        key: Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = key.py();
        self.inner_bind(py)
            .call_method1(intern!(py, "setdefault"), (key, default))
    }
}
impl ImplPyoReversible for Dict {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        self.inner_bind(py)
            .as_any()
            .pipe(pylibs::builtins::reversed)
            .into_pyochain()
    }
}
