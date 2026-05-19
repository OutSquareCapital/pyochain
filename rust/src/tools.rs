use crate::args::{Args, Concatenate, Kwargs};
use crate::option::{PySome, get_null};
use crate::result::{PyoErr, PyoOk};
use pyo3::types::{
    PyAny, PyBool, PyFunction, PyIterator, PyList, PyModule, PySequence, PySet, PyTuple,
};
use pyo3::{IntoPyObjectExt, prelude::*};
use pyo3::{ffi, intern};
/// Create a unique sentinel object
#[inline]
fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
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
    m.add_function(wrap_pyfunction!(try_for_each, m)?)?;
    m.add_function(wrap_pyfunction!(try_collect, m)?)?;
    m.add_function(wrap_pyfunction!(eq, m)?)?;
    m.add_function(wrap_pyfunction!(ne, m)?)?;
    m.add_function(wrap_pyfunction!(le, m)?)?;
    m.add_function(wrap_pyfunction!(lt, m)?)?;
    m.add_function(wrap_pyfunction!(gt, m)?)?;
    m.add_function(wrap_pyfunction!(ge, m)?)?;
    m.add_function(wrap_pyfunction!(all_unique, m)?)?;
    m.add_function(wrap_pyfunction!(all_unique_by, m)?)?;
    m.add_function(wrap_pyfunction!(partition, m)?)?;
    m.add_function(wrap_pyfunction!(last, m)?)?;
    m.add_function(wrap_pyfunction!(length, m)?)?;
    m.add_function(wrap_pyfunction!(retain, m)?)?;
    Ok(())
}
#[pyfunction]
#[pyo3(signature = (data, func, *args, **kwargs))]
pub fn for_each(
    data: &Bound<'_, PyAny>,
    func: &Bound<'_, PyAny>,
    args: &Args<'_>,
    kwargs: Option<&Kwargs<'_>>,
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
    data: Bound<'_, PyIterator>,
    func: Bound<'_, PyAny>,
    args: Args<'_>,
    kwargs: Option<&Kwargs<'_>>,
) -> PyResult<()> {
    match (args.is_empty(), kwargs) {
        (true, None) => data.try_iter()?.try_for_each(|item| {
            func.call1(item?.cast_exact::<PyTuple>()?)?;
            Ok(())
        }),
        (true, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.call(item?.cast_exact::<PyTuple>()?, kwargs)?;
            Ok(())
        }),
        (false, None) => data.try_iter()?.try_for_each(|item| {
            func.concat_star1(item?.cast_exact::<PyTuple>()?, &args)?;
            Ok(())
        }),
        (false, Some(_)) => data.try_iter()?.try_for_each(|item| {
            func.concat_star(item?.cast_exact::<PyTuple>()?, &args, kwargs)?;
            Ok(())
        }),
    }
}
#[pyfunction]
pub fn try_for_each(data: Bound<'_, PyIterator>, f: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    for item in data {
        let result = f.call1((&item?,))?;
        match result.cast_exact::<PyoOk>() {
            Ok(_) => (),
            Err(_) => return result.cast_exact::<PyoErr>()?.into_py_any(py),
        }
    }
    PyoOk::new(PyTuple::empty(py).into()).into_py_any(py)
}
#[pyfunction]
pub fn try_find(data: &Bound<'_, PyAny>, predicate: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    for item in data.try_iter()? {
        let val = item?;
        let result = predicate.call1((&val,))?;
        match result.cast_exact::<PyoOk>() {
            Ok(ok_ref) => {
                if unsafe {
                    ok_ref
                        .get()
                        .value
                        .cast_bound_unchecked::<PyBool>(py)
                        .is_true()
                } {
                    let some_val = PySome::new(val.unbind()).into_py_any(py)?;
                    return Ok(PyoOk::new(some_val).into_py_any(py)?);
                }
            }
            Err(_) => {
                return Ok(result
                    .cast_exact::<PyoErr>()?
                    .to_owned()
                    .unbind()
                    .into_any());
            }
        }
    }
    let none = get_null(py);
    Ok(PyoOk::new(none.into_py_any(py)?).into_py_any(py)?)
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
        let result = func.call1((accumulator, item))?;
        match result.cast_exact::<PyoOk>() {
            Ok(ok_ref) => {
                accumulator = ok_ref.get().value.clone_ref(py);
            }
            Err(_) => {
                return result.cast_exact::<PyoErr>()?.into_py_any(py);
            }
        }
    }
    return PyoOk::new(accumulator).into_py_any(py);
}

#[pyfunction]
pub fn try_reduce(data: &Bound<'_, PyAny>, func: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    let mut iterator = data.try_iter()?;
    let first = iterator.next();
    if first.is_none() {
        return Ok(PyoOk::new(get_null(py).into_py_any(py)?).into_py_any(py)?);
    }

    let mut accumulator = first.unwrap()?.to_owned().unbind();

    for item in iterator {
        let val = item?;
        let result = func.call1((&accumulator, val))?;
        match result.cast_exact::<PyoOk>() {
            Ok(ok_ref) => {
                accumulator = ok_ref.get().value.clone_ref(py);
            }
            Err(_) => {
                return result.cast_exact::<PyoErr>()?.into_py_any(py);
            }
        }
    }

    PyoOk::new(PySome::new(accumulator).into_py_any(py)?).into_py_any(py)
}
#[pyfunction]
pub fn try_collect(data: Bound<'_, PyIterator>) -> PyResult<Py<PyAny>> {
    let py = data.py();
    let collected = PyList::empty(py);

    for item in data {
        let val = item?;
        match val.cast_exact::<PyoOk>() {
            Ok(ok) => collected.append(&ok.get().value)?,
            Err(_) => match val.cast_into_exact::<PySome>() {
                Ok(some) => collected.append(&some.get().value)?,
                Err(_) => return get_null(py).into_py_any(py),
            },
        }
    }
    PySome::new(collected.into()).into_py_any(py)
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
#[pyfunction]
pub fn all_unique(mut data: Bound<'_, PyIterator>) -> PyResult<bool> {
    let seen = PySet::empty(data.py())?;
    while let Some(item) = data.next() {
        let key_value = item?;
        if seen.contains(&key_value)? {
            return Ok(false);
        }
        seen.add(key_value)?;
    }
    Ok(true)
}
#[pyfunction]
pub fn all_unique_by(data: Bound<'_, PyIterator>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut iter = data.map(|item| key.call1((item?,)));
    let seen = PySet::empty(key.py())?;
    while let Some(item) = iter.next() {
        let item = item?;
        if seen.contains(&item)? {
            return Ok(false);
        }
        seen.add(item)?;
    }
    Ok(true)
}
#[pyfunction]
pub fn partition(
    data: Bound<'_, PyIterator>,
    predicate: &Bound<'_, PyAny>,
) -> PyResult<(Py<PyList>, Py<PyList>)> {
    let py = data.py();
    let true_list = PyList::empty(py);
    let false_list = PyList::empty(py);
    for item in data {
        let item = item?;
        if predicate.call1((&item,))?.is_truthy()? {
            true_list.append(item)?;
        } else {
            false_list.append(item)?;
        }
    }
    Ok((true_list.unbind(), false_list.unbind()))
}
/// We use unsafe code here to match the performance of a Cython implementation
#[pyfunction]
pub fn last(data: Bound<'_, PyIterator>) -> PyResult<Py<PyAny>> {
    let py = data.py();

    // SAFETY:
    // - `data` is a valid `PyIterator`, so `iterator` is a valid `PyObject*` for the whole scope.
    // - The active `Python<'_>` token guarantees the GIL is held while calling CPython APIs.
    // - `tp_iternext` is the iterator next slot for this exact Python object type.
    // - Each successful `next(iterator)` returns a new owned reference which we either `Py_DECREF`
    //   or transfer to Python with `Py::from_owned_ptr`.
    // - On the error path after replacing `last`, we release the currently owned reference before
    //   fetching and returning the Python exception.
    unsafe {
        let iterator = data.as_ptr();
        let next = (*(*iterator).ob_type).tp_iternext.unwrap();

        let mut last = next(iterator);
        if last.is_null() {
            if ffi::PyErr_Occurred().is_null() {
                return Err(pyo3::exceptions::PyStopIteration::new_err(""));
            }
            return Err(PyErr::fetch(py));
        }

        loop {
            let item = next(iterator);
            if item.is_null() {
                if ffi::PyErr_Occurred().is_null() {
                    break;
                }
                ffi::Py_DECREF(last);
                return Err(PyErr::fetch(py));
            }
            ffi::Py_DECREF(last);
            last = item;
        }

        Ok(Py::from_owned_ptr(py, last))
    }
}
/// We use unsafe code here to match the performance of a Cython implementation
#[pyfunction]
pub fn length(data: Bound<'_, PyIterator>) -> PyResult<usize> {
    let py = data.py();
    let mut count = 0usize;
    let iterator = data.as_ptr();

    // SAFETY:
    // - `data` is a valid `PyIterator`, so `iterator` stays valid for the duration of the loop.
    // - The active `Python<'_>` token guarantees the GIL is held while calling CPython APIs.
    // - `tp_iternext` is the iterator next slot for this exact Python object type.
    // - Each non-null `item` is a new owned reference returned by CPython and is released exactly
    //   once with `Py_DECREF` after it has been counted.
    unsafe {
        let next = (*(*iterator).ob_type).tp_iternext.unwrap();
        loop {
            let item = next(iterator);
            if item.is_null() {
                if ffi::PyErr_Occurred().is_null() {
                    break;
                }
                return Err(PyErr::fetch(py));
            }
            count += 1;
            ffi::Py_DECREF(item);
        }
    }

    Ok(count)
}
#[pyfunction]
pub fn retain(data: Bound<'_, PySequence>, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
    let mut write_idx = 0;
    let length = data.len()?;
    for read_idx in 0..length {
        let curr = data.get_item(read_idx)?;
        if predicate.call1((&curr,))?.is_truthy()? {
            data.set_item(write_idx, curr)?;
            write_idx += 1;
        }
    }
    while data.len()? > write_idx {
        data.del_item(write_idx)?;
    }
    Ok(())
}
