/// Pure functions tools for pyochain
use crate::option::{PySome, get_none_singleton};
use crate::result::{PyErr as PyochainErr, PyOk, PyResultEnum};
use crate::types::PyClassInit;
use pyo3::intern;
use pyo3::types::{PyAny, PyBool, PyFunction, PyModule};
use pyo3::{IntoPyObjectExt, prelude::*};
/// Create a unique sentinel object
#[inline]
fn sentinel(py: Python<'_>) -> PyResult<Bound<PyAny>> {
    let sentinel = PyModule::import(py, "builtins")?
        .getattr(intern!(py, "object"))?
        .call0()?;
    Ok(sentinel)
}
#[inline]
fn zip_longest(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyModule::import(py, "itertools")?.getattr(intern!(py, "zip_longest"))?)
}

#[pymodule(name = "_tools")]
pub fn tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(try_find, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(ne, m)?)?;
    m.add_function(wrap_pyfunction!(le, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(ge, m)?)?;
    Ok(())
}
#[pyfunction]
pub fn try_find(data: &Bound<'_, PyAny>, predicate: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    for item in data.try_iter()? {
        let val = item?;
        match predicate.call1((&val,))?.extract::<PyResultEnum<'_>>()? {
            PyResultEnum::Ok(ok_ref) => {
                if unsafe {
                    ok_ref
                        .get()
                        .value
                        .cast_bound_unchecked::<PyBool>(py)
                        .is_true()
                } {
                    let some_val = PySome::new(val.unbind()).init(py)?.into_any();
                    return Ok(PyOk::new(some_val).into_py_any(py)?);
                }
            }
            PyResultEnum::Err(err_ref) => {
                let err_val = err_ref.get().error.clone_ref(py);
                return Ok(PyochainErr::new(err_val).into_py_any(py)?);
            }
        }
    }
    let none = get_none_singleton(py)?;
    Ok(PyOk::new(none).into_py_any(py)?)
}

#[pyfunction]
pub fn eq(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;

    for pair in zip_longest(py)?.call1((self_iter, other))?.try_iter()? {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) || b.is(&sentinel) || !a.eq(&b)? {
            return Ok(false);
        }
    }
    Ok(true)
}
#[pyfunction]
pub fn ne(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;
    for pair in zip_longest(py)?.call1((self_iter, other))?.try_iter()? {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) || b.is(&sentinel) || !a.eq(&b)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
pub fn le(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(intern!(py, "fillvalue"), &sentinel)?;
    for pair in zip_longest(py)?
        .call((self_iter, other), Some(&kwargs))?
        .try_iter()?
    {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) {
            return Ok(true);
        }
        if b.is(&sentinel) {
            return Ok(false);
        }
        if !a.eq(&b)? {
            return Ok(a.lt(&b)?);
        }
    }
    Ok(true)
}

#[pyfunction]
pub fn lt(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(intern!(py, "fillvalue"), &sentinel)?;
    for pair in zip_longest(py)?
        .call((self_iter, other), Some(&kwargs))?
        .try_iter()?
    {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) {
            return Ok(true);
        }
        if b.is(&sentinel) {
            return Ok(false);
        }
        if !a.eq(&b)? {
            return Ok(a.lt(&b)?);
        }
    }
    Ok(false)
}

#[pyfunction]
pub fn gt(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(intern!(py, "fillvalue"), &sentinel)?;
    for pair in zip_longest(py)?
        .call((self_iter, other), Some(&kwargs))?
        .try_iter()?
    {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) {
            return Ok(false);
        }
        if b.is(&sentinel) {
            return Ok(true);
        }
        if !a.eq(&b)? {
            return Ok(a.gt(&b)?);
        }
    }
    Ok(false)
}

#[pyfunction]
pub fn ge(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;
    let kwargs = pyo3::types::PyDict::new(py);
    kwargs.set_item(intern!(py, "fillvalue"), &sentinel)?;
    for pair in zip_longest(py)?
        .call((self_iter, other), Some(&kwargs))?
        .try_iter()?
    {
        let (a, b) = pair?.extract::<(Bound<PyAny>, Bound<PyAny>)>()?;
        if a.is(&sentinel) {
            return Ok(false);
        }
        if b.is(&sentinel) {
            return Ok(true);
        }
        if !a.eq(&b)? {
            return Ok(a.gt(&b)?);
        }
    }
    Ok(true)
}
