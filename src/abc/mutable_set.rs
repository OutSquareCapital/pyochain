use pyo3::{exceptions::PyKeyError, intern, prelude::*};
use pyo3_ext::{
    prelude::*,
    types::{PopResult, PyAbstractSet, PyMutableSet},
};
use pyochain_macros::try_cast;

use crate::abc::{PyoSet, set::py_from_iterable};
#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSet)]
pub struct PyoMutableSet;

#[pymethods]
impl PyoMutableSet {
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
                    Case::PyAbstractSet(x) => py_from_iterable(&slf, x)?.into_any(),
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
        new_pop_result(slf.as_any())
            .into_pyresult()
            .and_then(|value| {
                Self::into_mutable_set(slf).discard(&value)?;
                Ok(value)
            })
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let any = slf.as_any();
        let slf = Self::as_mutable_set(&slf);
        loop {
            match new_pop_result(any) {
                PopResult::Err(e) => {
                    return Err(e);
                }
                PopResult::KeyMissing => break Ok(()),
                PopResult::Ok(v) => {
                    slf.discard(&v)?;
                    continue;
                }
            }
        }
    }
}
#[inline]
fn new_pop_result<'py>(slf: &Bound<'py, PyAny>) -> PopResult<'py> {
    slf.try_iter()
        .map(|mut x| match x.next() {
            None => PopResult::KeyMissing,
            Some(Ok(v)) => PopResult::Ok(v),
            Some(Err(e)) => PopResult::Err(e),
        })
        .unwrap_or_else(PopResult::Err)
}
