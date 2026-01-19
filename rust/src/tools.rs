/// Pure functions tools for pyochain
use crate::option::{PySome, get_none_singleton};
use crate::result::{PyOk, PyResultEnum};
use crate::types::{ConcatArgs, PyClassInit};
use pyo3::intern;
use pyo3::types::{PyAny, PyBool, PyDict, PyFunction, PyModule, PyTuple};
use pyo3::{IntoPyObjectExt, prelude::*};

mod itertools {
    use super::*;
    #[inline]
    fn import(py: Python<'_>) -> PyResult<Bound<PyModule>> {
        PyModule::import(py, intern!(py, "itertools"))
    }
    #[inline]
    pub fn groupby(py: Python<'_>) -> PyResult<Bound<PyAny>> {
        import(py)?.getattr(intern!(py, "groupby"))
    }
}
/// Create a unique sentinel object
#[inline]
fn sentinel(py: Python<'_>) -> PyResult<Bound<PyAny>> {
    let sentinel = PyModule::import(py, intern!(py, "builtins"))?
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
    m.add_function(wrap_pyfunction!(for_each, m)?)?;
    m.add_function(wrap_pyfunction!(for_each_star, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(ne, m)?)?;
    m.add_function(wrap_pyfunction!(le, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(ge, m)?)?;
    m.add_function(wrap_pyfunction!(all_equal, m)?)?;
    m.add_function(wrap_pyfunction!(all_equal_by, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (data, func, *args, **kwargs))]
pub fn for_each(
    data: &Bound<'_, PyAny>,
    func: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    match (args.is_empty(), kwargs) {
        (true, None) => data.try_iter()?.try_for_each(|item| {
            func.call1((&item?,))?;
            Ok(())
        }),
        (true, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.call((&item?,), kwargs)?;
            Ok(())
        }),
        (false, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.concat(&item?, args, kwargs)?;
            Ok(())
        }),
        (false, None) => data.try_iter()?.try_for_each(|item| {
            func.concat1(&item?, args)?;
            Ok(())
        }),
    }
}
#[pyfunction]
#[pyo3(signature = (data, func, *args, **kwargs))]
pub fn for_each_star(
    data: &Bound<'_, PyAny>,
    func: &Bound<'_, PyAny>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<()> {
    match (args.is_empty(), kwargs) {
        (true, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.call(&item?.cast_into::<PyTuple>()?, kwargs)?;
            Ok(())
        }),
        (true, None) => data.try_iter()?.try_for_each(|item| {
            func.call1(item?.cast_into::<PyTuple>()?)?;
            Ok(())
        }),
        (false, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.concat_star(&item?.cast_into::<PyTuple>()?, args, kwargs)?;
            Ok(())
        }),
        (false, None) => data.try_iter()?.try_for_each(|item| {
            func.concat_star1(&item?.cast_into::<PyTuple>()?, args)?;
            Ok(())
        }),
    }
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
    Ok(PyOk::new(get_none_singleton(py)?).into_py_any(py)?)
}
#[pyfunction]
pub fn try_fold(
    data: &Bound<'_, PyAny>,
    init: &Bound<'_, PyAny>,
    func: &Bound<'_, PyFunction>,
) -> PyResult<Py<PyAny>> {
    let py = data.py();
    let mut accumulator = init.to_owned().unbind();

    for item in data.try_iter()? {
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
pub fn try_reduce(data: &Bound<'_, PyAny>, func: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    let mut iterator = data.try_iter()?;
    let first = iterator.next();
    if first.is_none() {
        return Ok(PyOk::new(get_none_singleton(py)?).into_py_any(py)?);
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
    data: &Bound<'_, PyAny>,
    reverse: &Bound<'_, PyBool>,
    strict: &Bound<'_, PyBool>,
) -> PyResult<bool> {
    let mut iter = data.try_iter()?;
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
    data: &Bound<'_, PyAny>,
    key: &Bound<'_, PyAny>,
    reverse: &Bound<'_, PyBool>,
    strict: &Bound<'_, PyBool>,
) -> PyResult<bool> {
    let mut iter = data.try_iter()?;
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
pub fn eq(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let py = data.py();
    let sentinel = sentinel(py)?;

    let mut left_iter = data.try_iter()?;
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
pub fn ne(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = data.try_iter()?;
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
pub fn le(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = data.try_iter()?;
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
pub fn lt(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = data.try_iter()?;
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
pub fn gt(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = data.try_iter()?;
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
pub fn ge(data: &Bound<'_, PyAny>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut left_iter = data.try_iter()?;
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

/// Enum to hold the extracted first value type for specialization
enum FirstValue {
    Int(i64),
    Bool(bool),
    Other,
}
#[pyfunction]
pub fn all_equal(data: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut iter = data.try_iter()?;
    let Some(first) = iter.next() else {
        return Ok(true);
    };
    let first = first?;

    let first_val = match first.extract::<i64>() {
        Ok(v) => FirstValue::Int(v),
        Err(_) => match first.extract::<bool>() {
            Ok(v) => FirstValue::Bool(v),
            Err(_) => FirstValue::Other,
        },
    };

    match first_val {
        FirstValue::Int(first_int) => {
            for item in iter {
                let item = item?;
                match item.extract::<i64>() {
                    Ok(val) => {
                        if val != first_int {
                            return Ok(false);
                        }
                    }
                    Err(_) => {
                        return Ok(false);
                    }
                }
            }
        }
        FirstValue::Bool(first_bool) => {
            for item in iter {
                let item = item?;
                match item.extract::<bool>() {
                    Ok(val) => {
                        if val != first_bool {
                            return Ok(false);
                        }
                    }
                    Err(_) => {
                        return Ok(false);
                    }
                }
            }
        }
        FirstValue::Other => {
            let iterator = itertools::groupby(data.py())?.call1((data,))?.try_iter()?;
            for _first in &iterator {
                for _second in &iterator {
                    return Ok(false);
                }
                return Ok(true);
            }
        }
    }
    Ok(true)
}

#[pyfunction]
pub fn all_equal_by(data: &Bound<'_, PyAny>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut iter = data.try_iter()?;
    let Some(first) = iter.next() else {
        return Ok(true);
    };
    let first_key = key.call1((first?,))?;

    for item in iter {
        if !first_key.eq(&key.call1((item?,))?)? {
            return Ok(false);
        }
    }
    Ok(true)
}
