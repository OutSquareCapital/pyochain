use crate::args::{Args, Concatenate, Kwargs};
use crate::option::{PyNull, PySome};
use crate::result::{PyoErr, PyoOk};
use pyo3::types::{
    PyAny, PyBool, PyDict, PyFunction, PyIterator, PyList, PyModule, PySequence, PySet, PyTuple,
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
    m.add_class::<UniqueIdentity>()?;
    m.add_class::<UniqueKey>()?;
    m.add_class::<Intersperse>()?;
    m.add_class::<SlidingWindow>()?;
    m.add_class::<Juxt>()?;
    m.add_class::<FilterMap>()?;
    m.add_class::<FilterMapStar>()?;
    m.add_class::<Scan>()?;
    m.add_class::<MapWhile>()?;
    m.add_class::<FromFn>()?;
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
    let none = PyNull::get(py);
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
        return Ok(PyoOk::new(PyNull::get(py).into_py_any(py)?).into_py_any(py)?);
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
                Err(_) => return PyNull::get(py).into_py_any(py),
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
            if ffi::PyErr_ExceptionMatches(ffi::PyExc_StopIteration) != 0 {
                ffi::PyErr_Clear();
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
                if ffi::PyErr_ExceptionMatches(ffi::PyExc_StopIteration) != 0 {
                    ffi::PyErr_Clear();
                    break;
                }
                ffi::Py_DECREF(last);
                return Err(PyErr::fetch(py));
            }
            ffi::Py_DECREF(last);
            last = item;
        }

        Ok(Bound::from_owned_ptr(py, last).unbind())
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
                if ffi::PyErr_ExceptionMatches(ffi::PyExc_StopIteration) != 0 {
                    ffi::PyErr_Clear();
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
#[pyclass]
pub struct Juxt {
    funcs: Vec<Py<PyAny>>,
}

#[pymethods]
impl Juxt {
    #[new]
    #[pyo3(signature = (*funcs))]
    fn new(funcs: &Bound<'_, PyTuple>) -> PyResult<Self> {
        let collected = funcs
            .try_iter()?
            .map(|item| item.map(Bound::unbind))
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self { funcs: collected })
    }

    #[pyo3(signature = (*args))]
    fn __call__(&self, args: &Bound<'_, PyTuple>) -> PyResult<Py<PyTuple>> {
        let py = args.py();
        let results = self
            .funcs
            .iter()
            .map(|func| func.call1(py, args))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, results)?.unbind())
    }
}
/// TODO: speed is 0.76x compared to the Cython implementation.
/// Saved in `.benchmarks/unique_cy`
#[pyclass(frozen)]
pub struct UniqueIdentity {
    iter: Py<PyIterator>,
    seen: Py<PySet>,
}

#[pymethods]
impl UniqueIdentity {
    #[new]
    fn new(data: Bound<'_, PyIterator>) -> PyResult<Self> {
        let py = data.py();
        Ok(Self {
            iter: data.unbind(),
            seen: PySet::empty(py)?.unbind(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        let seen = slf.seen.bind(py);

        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let item = result?;
                    if seen.contains(&item)? {
                        continue;
                    }
                    seen.add(&item)?;
                    return Ok(Some(item));
                }
            }
        }
    }
}
/// TODO: speed is 0.95x compared to the Cython implementation.
/// Saved in `.benchmarks/unique_cy`
#[pyclass(frozen)]
pub struct UniqueKey {
    iter: Py<PyIterator>,
    key: Py<PyAny>,
    seen: Py<PySet>,
}

#[pymethods]
impl UniqueKey {
    #[new]
    fn new(data: Bound<'_, PyIterator>, key: Bound<'_, PyAny>) -> PyResult<Self> {
        let py = data.py();
        Ok(Self {
            iter: data.unbind(),
            key: key.unbind(),
            seen: PySet::empty(py)?.unbind(),
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRef<'_, Self>) -> PyResult<Option<Bound<'_, PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        let key = slf.key.bind(py);
        let seen = slf.seen.bind(py);

        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let item = result?;
                    let tag = key.call1((&item,))?;
                    if seen.contains(&tag)? {
                        continue;
                    }
                    seen.add(&tag)?;
                    return Ok(Some(item));
                }
            }
        }
    }
}
/// TODO: speed is 0.44x compared to the Cython implementation.
/// Saved in `.benchmarks/intersperse_cy`
/// Cytoolz median time in us:
/// 256 elements: 10.6
/// 1024 elements: 37.3
/// 4096 elements: 127.7
#[pyclass]
pub struct Intersperse {
    data: Py<PyIterator>,
    element: Py<PyAny>,
    val: Option<Py<PyAny>>,
    must_process: bool,
}

#[pymethods]
impl Intersperse {
    #[new]
    fn new(mut data: Bound<'_, PyIterator>, element: Py<PyAny>) -> PyResult<Self> {
        let (val, must_process) = match data.next() {
            None => (None, true),
            Some(item) => (Some(item?.unbind()), false),
        };
        Ok(Self {
            data: data.unbind(),
            element,
            val,
            must_process,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        if slf.must_process {
            match slf.data.clone_ref(py).into_bound(py).next() {
                None => Ok(None),
                Some(item) => {
                    slf.val = Some(item?.unbind());
                    slf.must_process = false;
                    Ok(Some(slf.element.clone_ref(py)))
                }
            }
        } else {
            slf.must_process = true;
            Ok(slf.val.as_ref().map(|v| v.clone_ref(py)))
        }
    }
}
///TODO: It's actually slower than cytoolz implementation when `n` is small, we should optimize for that case.\
/// Observed speeds:\
/// **0.81x** -> `n=2`\
/// **0.93x** -> `n=8`\
/// **1.17x** -> `n=32`\
/// **1.40x** -> `n=128`\
#[pyclass]
pub struct SlidingWindow {
    iter: Py<PyIterator>,
    prev: Vec<Py<PyAny>>,
}

#[pymethods]
impl SlidingWindow {
    #[new]
    fn new(mut data: Bound<'_, PyIterator>, n: usize) -> PyResult<Self> {
        let py = data.py();
        let mut prev: Vec<Py<PyAny>> = (0..n).map(|_| py.None().into_any()).collect();
        for i in 1..n {
            match data.next() {
                None => break,
                Some(item) => prev[i] = item?.unbind(),
            }
        }
        Ok(Self {
            iter: data.unbind(),
            prev,
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyTuple>>> {
        let py = slf.py();
        let item = match slf.iter.clone_ref(py).into_bound(py).next() {
            None => return Ok(None),
            Some(result) => result?.unbind(),
        };
        slf.prev.rotate_left(1);
        let last = slf.prev.len() - 1;
        slf.prev[last] = item;
        let tuple = PyTuple::new(py, slf.prev.iter())?;
        Ok(Some(tuple.into()))
    }
}
#[pyclass]
pub struct FilterMap {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl FilterMap {
    #[new]
    fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            iter: data.unbind(),
            func: func.unbind(),
        })
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let res = func.call1((result?,))?;
                    match res.cast_into_exact::<PySome>() {
                        Ok(some) => return Ok(Some(some.get().value.clone_ref(py))),
                        Err(_) => continue,
                    }
                }
            }
        }
    }
}
#[pyclass]
pub struct FilterMapStar {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl FilterMapStar {
    #[new]
    fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self {
            iter: data.unbind(),
            func: func.unbind(),
        })
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        loop {
            match iter.next() {
                None => return Ok(None),
                Some(result) => {
                    let res = func.call1(result?.cast_exact::<PyTuple>()?)?;
                    match res.cast_into_exact::<PySome>() {
                        Ok(some) => return Ok(Some(some.get().value.clone_ref(py))),
                        Err(_) => continue,
                    }
                }
            }
        }
    }
}

#[pyclass]
pub struct Scan {
    iter: Py<PyIterator>,
    initial: Py<PyAny>,
    func: Py<PyAny>,
}
#[pymethods]
impl Scan {
    #[new]
    fn new(
        data: Bound<'_, PyIterator>,
        initial: Bound<'_, PyAny>,
        func: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            iter: data.unbind(),
            initial: initial.unbind(),
            func: func.unbind(),
        })
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let func = slf.func.bind(py);
        let mut iter = slf.iter.clone_ref(py).into_bound(py);

        match iter.next() {
            None => Ok(None),
            Some(result) => {
                let state = slf.initial.clone_ref(py);
                let res = func.call1((state, result?))?;

                match res.cast_exact::<PySome>() {
                    Ok(some) => {
                        let next_state = some.get().value.clone_ref(py);
                        slf.initial = next_state.clone_ref(py);
                        Ok(Some(next_state))
                    }
                    Err(_) => Ok(None),
                }
            }
        }
    }
}

#[pyclass]
pub struct MapWhile {
    iter: Py<PyIterator>,
    func: Py<PyAny>,
}
#[pymethods]
impl MapWhile {
    #[new]
    fn new(data: Bound<'_, PyIterator>, func: Bound<'_, PyAny>) -> Self {
        Self {
            iter: data.unbind(),
            func: func.unbind(),
        }
    }
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let mut iter = slf.iter.clone_ref(py).into_bound(py);
        match iter.next() {
            None => Ok(None),
            Some(result) => match slf.func.bind(py).call1((result?,))?.cast_exact::<PySome>() {
                Ok(some) => Ok(Some(some.get().value.clone_ref(py))),
                Err(_) => Ok(None),
            },
        }
    }
}

enum FromFnStrategy {
    NoArgs,
    HasArgs(Py<PyTuple>),
    HasKwargs(Py<PyDict>),
    HasBoth(Py<PyTuple>, Py<PyDict>),
}
impl FromFnStrategy {
    fn new(args: &Args<'_>, kwargs: Option<&Kwargs<'_>>) -> Self {
        match (args.is_empty(), kwargs) {
            (true, None) => Self::NoArgs,
            (false, None) => Self::HasArgs(args.to_owned().unbind()),
            (true, Some(kwargs)) => Self::HasKwargs(kwargs.to_owned().unbind()),
            (false, Some(kwargs)) => {
                Self::HasBoth(args.to_owned().unbind(), kwargs.to_owned().unbind())
            }
        }
    }
}
#[pyclass]
pub struct FromFn {
    func: Py<PyAny>,
    strategy: FromFnStrategy,
}
#[pymethods]
impl FromFn {
    #[pyo3(signature = (func, *args, **kwargs))]
    #[new]
    fn new(func: Bound<'_, PyAny>, args: &Args<'_>, kwargs: Option<&Kwargs<'_>>) -> Self {
        Self {
            func: func.unbind(),
            strategy: FromFnStrategy::new(args, kwargs),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(slf: PyRefMut<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let py_fn = slf.func.bind(py);
        let res = match slf.strategy {
            FromFnStrategy::NoArgs => py_fn.call0()?,
            FromFnStrategy::HasArgs(ref args) => py_fn.call1(args.bind(py))?,
            FromFnStrategy::HasKwargs(ref kwargs) => py_fn.call((), Some(kwargs.bind(py)))?,
            FromFnStrategy::HasBoth(ref args, ref kwargs) => {
                py_fn.call(args.bind(py), Some(kwargs.bind(py)))?
            }
        };
        match res.cast_into_exact::<PySome>() {
            Ok(some) => Ok(Some(some.get().value.clone_ref(py))),
            Err(_) => Ok(None),
        }
    }
}
