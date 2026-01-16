/// Pure functions tools for pyochain
use crate::option::{PySome, get_none_singleton};
use crate::result::{PyOk, PyResultEnum};
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
#[pymodule(name = "_tools")]
pub fn tools(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(try_find, m)?)?;
    m.add_function(wrap_pyfunction!(try_fold, m)?)?;
    m.add_function(wrap_pyfunction!(try_reduce, m)?)?;
    m.add_function(wrap_pyfunction!(is_sorted, m)?)?;
    m.add_function(wrap_pyfunction!(is_sorted_by, m)?)?;
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
                return Ok(err_ref.into_py_any(py)?);
            }
        }
    }
    let none = get_none_singleton(py)?;
    Ok(PyOk::new(none).into_py_any(py)?)
}
#[pyfunction]
pub fn try_fold(
    self_iter: &Bound<'_, PyAny>,
    init: &Bound<'_, PyAny>,
    func: &Bound<'_, PyFunction>,
) -> PyResult<Py<PyAny>> {
    let py = self_iter.py();
    let mut accumulator = init.to_owned().unbind();

    for item in self_iter.try_iter()? {
        let item = item?;
        match func
            .call1((accumulator, item))?
            .extract::<PyResultEnum<'_>>()?
        {
            PyResultEnum::Ok(ok_ref) => {
                accumulator = ok_ref.get().value.clone_ref(py);
            }
            PyResultEnum::Err(err_ref) => {
                return Ok(err_ref.into_py_any(py)?);
            }
        }
    }
    return Ok(PyOk::new(accumulator).into_py_any(py)?);
}

#[pyfunction]
pub fn try_reduce(
    self_iter: &Bound<'_, PyAny>,
    func: &Bound<'_, PyFunction>,
) -> PyResult<Py<PyAny>> {
    let py = self_iter.py();
    let mut iterator = self_iter.try_iter()?;
    let first = iterator.next();
    if first.is_none() {
        return Ok(get_none_singleton(py)?.into_py_any(py)?);
    }

    let mut accumulator = first.unwrap()?.to_owned().unbind();

    for item in iterator {
        let val = item?;
        match func
            .call1((&accumulator, val))?
            .extract::<PyResultEnum<'_>>()?
        {
            PyResultEnum::Ok(ok_ref) => {
                accumulator = ok_ref.get().value.clone_ref(py);
            }
            PyResultEnum::Err(err_ref) => {
                return Ok(err_ref.into_py_any(py)?);
            }
        }
    }

    Ok(PyOk::new(PySome::new(accumulator).init(py)?.into_any()).into_py_any(py)?)
}
#[pyfunction]
pub fn is_sorted(
    self_iter: &Bound<'_, PyAny>,
    reverse: &Bound<'_, PyBool>,
    strict: &Bound<'_, PyBool>,
) -> PyResult<bool> {
    let mut iter = self_iter.try_iter()?;
    let Some(first) = iter.next() else {
        return Ok(true);
    };
    let mut prev = first?;

    match (strict.is_true(), reverse.is_true()) {
        (true, false) => {
            for item in iter {
                let curr = item?;
                if !prev.lt(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (false, false) => {
            for item in iter {
                let curr = item?;
                if !prev.le(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (true, true) => {
            for item in iter {
                let curr = item?;
                if !prev.gt(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (false, true) => {
            for item in iter {
                let curr = item?;
                if !prev.ge(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
    }
    Ok(true)
}
#[pyfunction]
pub fn is_sorted_by(
    self_iter: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    reverse: &Bound<'_, PyBool>,
    strict: &Bound<'_, PyBool>,
) -> PyResult<bool> {
    let mut iter = self_iter.try_iter()?;
    let Some(first) = iter.next() else {
        return Ok(true);
    };
    let mut prev = key.call1((first?,))?;
    match (strict.is_true(), reverse.is_true()) {
        (true, false) => {
            for item in iter {
                let curr = key.call1((item?,))?;
                if !prev.lt(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (false, false) => {
            for item in iter {
                let curr = key.call1((item?,))?;
                if !prev.le(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (true, true) => {
            for item in iter {
                let curr = key.call1((item?,))?;
                if !prev.gt(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
        (false, true) => {
            for item in iter {
                let curr = key.call1((item?,))?;
                if !prev.ge(&curr)? {
                    return Ok(false);
                }
                prev = curr;
            }
        }
    }
    Ok(true)
}

#[pyfunction]
pub fn eq(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = self_iter.py();
    let sentinel = sentinel(py)?;

    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if left.is(&sentinel) || right.is(&sentinel) || !left.eq(&right)? {
                    return Ok(false);
                }
            }
            (None, None) => return Ok(true),
            _ => return Ok(false),
        }
    }
}
#[pyfunction]
pub fn ne(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if !left.eq(&right)? {
                    return Ok(true);
                }
            }
            (None, None) => return Ok(false),
            _ => return Ok(true),
        }
    }
}
#[pyfunction]
pub fn le(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if !left.eq(&right)? {
                    return Ok(left.lt(&right)?);
                }
            }
            (None, None) => return Ok(true),
            (None, Some(_)) => return Ok(true),
            (Some(_), None) => return Ok(false),
        }
    }
}
#[pyfunction]
pub fn lt(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if !left.eq(&right)? {
                    return Ok(left.lt(&right)?);
                }
            }
            (None, None) => return Ok(false),
            (None, Some(_)) => return Ok(true),
            (Some(_), None) => return Ok(false),
        }
    }
}
#[pyfunction]
pub fn gt(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if !left.eq(&right)? {
                    return Ok(left.gt(&right)?);
                }
            }
            (None, None) => return Ok(false),
            (None, Some(_)) => return Ok(false),
            (Some(_), None) => return Ok(true),
        }
    }
}
#[pyfunction]
pub fn ge(self_iter: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = self_iter.try_iter()?;
    let mut right_iter = other.try_iter()?;

    loop {
        match (left_iter.next(), right_iter.next()) {
            (Some(left_res), Some(right_res)) => {
                let left = left_res?;
                let right = right_res?;
                if !left.eq(&right)? {
                    return Ok(left.gt(&right)?);
                }
            }
            (None, None) => return Ok(true),
            (None, Some(_)) => return Ok(false),
            (Some(_), None) => return Ok(true),
        }
    }
}
