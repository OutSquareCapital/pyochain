use pyo3::{exceptions::PyKeyError, intern, prelude::*};
use pyo3_ext::{
    prelude::*,
    types::{PyAbstractSet, PyMutableSet},
};
use pyochain_macros::try_cast;

use crate::{
    abc::{PyoSet, set::py_from_iterable},
    traits::PyoABC,
};
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSet)]
pub struct PyoMutableSet;

#[pymethods]
impl PyoMutableSet {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn into_mutable_set(slf: Bound<'_, Self>) -> Bound<'_, PyMutableSet> {
        unsafe { slf.cast_into_unchecked::<PyMutableSet>() }
    }

    fn as_mutable_set<'a, 'py>(slf: &'a Bound<'py, Self>) -> &'a Bound<'py, PyMutableSet> {
        unsafe { slf.cast_unchecked::<PyMutableSet>() }
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let slf = Self::into_mutable_set(slf);
        it.try_iter()?.try_for_each(|value| slf.add(&value?))
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let slf = Self::into_mutable_set(slf);
        slf.sub(&it)?
            .try_iter()?
            .try_for_each(|value| slf.discard(&value?))
    }

    fn __isub__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let slf = Self::into_mutable_set(slf);
            for value in it.try_iter()? {
                slf.discard(&value?)?;
            }
        }
        Ok(())
    }

    fn __ixor__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let pyset = try_cast! {
                match it {
                    Case::PyAbstractSet(x) => py_from_iterable(&slf, &x)?.into_any(),
                    iterable => iterable,
                }
            };
            let slf = Self::into_mutable_set(slf);
            for value in pyset.try_iter()? {
                let v = value?;
                if slf.contains(&v)? {
                    slf.discard(&v)?;
                } else {
                    slf.add(&v)?;
                }
            }
        }

        Ok(())
    }

    fn remove(slf: Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if !slf.contains(&value)? {
            Err(PyKeyError::new_err(format!("{}", value)))
        } else {
            Self::into_mutable_set(slf).discard(&value)?;
            Ok(())
        }
    }

    fn pop(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyAny>> {
        match PopResult::new(slf.as_any()) {
            PopResult::KeyStop => Err(PyKeyError::new_err("")),
            PopResult::Value(value) => {
                Self::into_mutable_set(slf).discard(&value)?;
                Ok(value)
            }
            PopResult::Error(e) => Err(e),
        }
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let any = slf.as_any();
        let slf = Self::as_mutable_set(&slf);
        loop {
            match PopResult::new(any) {
                PopResult::Error(e) => {
                    return Err(e);
                }
                PopResult::KeyStop => break Ok(()),
                PopResult::Value(v) => {
                    slf.discard(&v)?;
                    continue;
                }
            }
        }
    }
}
enum PopResult<'py> {
    Value(Bound<'py, PyAny>),
    KeyStop,
    Error(PyErr),
}
impl<'py> PopResult<'py> {
    fn new(slf: &Bound<'py, PyAny>) -> Self {
        slf.try_iter()
            .map(|mut x| match x.next() {
                None => PopResult::KeyStop,
                Some(Ok(v)) => PopResult::Value(v),
                Some(Err(e)) => PopResult::Error(e),
            })
            .unwrap_or_else(|e| PopResult::Error(e))
    }
}
