use crate::args::{Args, Concatenate, Kwargs};
use crate::mixins::Checkable;
use crate::option::{PyNull, PySome};
use crate::pylibs;
use crate::result::{PyoErr, PyoOk};
use crate::tools as tls;
use pyo3::exceptions::{PyStopIteration, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFunction, PyIterator, PyList, PySet, PyTuple, PyType};
use pyo3::{BoundObject, IntoPyObjectExt, ffi};
use tap::prelude::*;
#[pymodule(name = "_abc")]
pub fn abc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyoIterable>()?;
    m.add_class::<PyoIterator>()?;
    Ok(())
}
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(PyoIterable {})
    }
    fn iter<'py>(slf: &'py Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        pylibs::pyochain::iter::new(slf)
    }
}
///TODO: once the migration is done, rename it to `PyoIterator`
#[pyclass(name="PyoIteratorRS", subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoIterator;

#[pymethods]
impl PyoIterator {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoIterable {})
            .add_subclass(PyoIterator {})
    }
    #[classmethod]
    fn _from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        iterable: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((iterable,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (f, *args, **kwargs))]
    #[classmethod]
    fn from_fn<'py>(
        cls: &Bound<'py, PyType>,
        f: Bound<'py, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((tls::FromFn::new(f, args, kwargs),))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[classmethod]
    fn successors<'py>(
        cls: &Bound<'py, PyType>,
        first: Bound<'py, PyAny>,
        succ: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((tls::Successors::new(first, succ),))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[classmethod]
    #[pyo3(signature = (start=0, step=1))]
    fn from_count<'py>(
        cls: &Bound<'py, PyType>,
        start: i32,
        step: i32,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((pylibs::itertools::count(cls.py(), &start, &step)?,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    /// We use unsafe code here to match the performance of a Cython implementation
    fn last(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();

        // SAFETY:
        // - `slf` is a valid `PyIterator`, so `iterator` is a valid `PyObject*` for the whole scope.
        // - The active `Python<'_>` token guarantees the GIL is held while calling CPython APIs.
        // - `tp_iternext` is the iterator next slot for this exact Python object type.
        // - Each successful `next(iterator)` returns a new owned reference which we either `Py_DECREF`
        //   or transfer to Python with `Py::from_owned_ptr`.
        // - On the error path after replacing `last`, we release the currently owned reference before
        //   fetching and returning the Python exception.
        unsafe {
            let iterator = slf.as_ptr();
            let next = (*(*iterator).ob_type)
                .tp_iternext
                .expect("Iterator does not have tp_iternext");

            let mut last = next(iterator);
            if last.is_null() {
                if ffi::PyErr_Occurred().is_null() {
                    return Err(PyStopIteration::new_err(""));
                }
                if ffi::PyErr_ExceptionMatches(ffi::PyExc_StopIteration) != 0 {
                    ffi::PyErr_Clear();
                    return Err(PyStopIteration::new_err(""));
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

            Bound::from_owned_ptr(py, last).unbind().pipe(Ok)
        }
    }
    /// We use unsafe code here to match the performance of a Cython implementation
    fn count(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        let mut count = 0usize;
        let iterator = slf.as_ptr();

        // SAFETY:
        // - `slf` is a valid `PyIterator`, so `iterator` stays valid for the duration of the loop.
        // - The active `Python<'_>` token guarantees the GIL is held while calling CPython APIs.
        // - `tp_iternext` is the iterator next slot for this exact Python object type.
        // - Each non-null `item` is a new owned reference returned by CPython and is released exactly
        //   once with `Py_DECREF` after it has been counted.
        unsafe {
            let next = (*(*iterator).ob_type)
                .tp_iternext
                .expect("Iterator does not have tp_iternext");
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

                ffi::Py_DECREF(item);
                count += 1;
            }
        }

        Ok(count)
    }

    #[pyo3(signature = (predicate=None))]
    fn any<'py>(
        slf: &Bound<'py, Self>,
        predicate: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBool>> {
        let mut slf = slf.try_iter()?;
        match predicate {
            Some(pred) => slf
                .any(|item| {
                    item.and_then(|it| pred.call1((it,)))
                        .and_then(|res| res.is_truthy())
                        .expect("Error occurred while evaluating predicate in `any`")
                })
                .pipe(|x| PyBool::new(slf.py(), x))
                .into_bound()
                .pipe(Ok),
            None => pylibs::builtins::any(slf),
        }
    }
    #[pyo3(signature = (predicate=None))]
    fn all<'py>(
        slf: &Bound<'py, Self>,
        predicate: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBool>> {
        let mut slf = slf.try_iter()?;
        match predicate {
            Some(pred) => slf
                .all(|item| {
                    item.and_then(|it| pred.call1((it,)))
                        .and_then(|res| res.is_truthy())
                        .expect("Error occurred while evaluating predicate in `all`")
                })
                .pipe(|x| PyBool::new(slf.py(), x))
                .into_bound()
                .pipe(Ok),
            None => pylibs::builtins::all(slf),
        }
    }

    fn arg_min(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let mut slf = slf.try_iter()?;
        match slf.next() {
            None => Err(PyValueError::new_err(
                "Cannot compute `PyoIterator::arg_min` of an empty Iterator",
            )),
            Some(first) => {
                let mut best_index = 0;
                let mut best_value = first?;

                slf.enumerate().try_for_each(|(index, item)| {
                    let value = item?;
                    if value.lt(&best_value)? {
                        best_index = index + 1;
                        best_value = value;
                    }
                    Ok::<(), PyErr>(())
                })?;

                Ok(best_index)
            }
        }
    }
    fn arg_max(slf: &Bound<'_, Self>) -> PyResult<usize> {
        let mut slf = slf.try_iter()?;
        match slf.next() {
            None => Err(PyValueError::new_err(
                "Cannot compute `PyoIterator::arg_max` of an empty Iterator",
            )),
            Some(first) => {
                let mut best_index = 0;
                let mut best_value = first?;

                slf.enumerate().try_for_each(|(index, item)| {
                    let value = item?;
                    if value.gt(&best_value)? {
                        best_index = index + 1;
                        best_value = value;
                    }
                    Ok::<(), PyErr>(())
                })?;

                Ok(best_index)
            }
        }
    }
    fn arg_min_by(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut slf = slf.try_iter()?;
        match slf.next() {
            None => Err(PyValueError::new_err(
                "Cannot compute `PyoIterator::arg_min_by` of an empty Iterator",
            )),
            Some(first) => {
                let mut best_index = 0;
                let mut best_value = key.call1((first?,))?;

                slf.map(|x| key.call1((x?,)))
                    .enumerate()
                    .try_for_each(|(index, item)| {
                        let value = item?;
                        if value.lt(&best_value)? {
                            best_index = index + 1;
                            best_value = value;
                        }

                        Ok::<(), PyErr>(())
                    })?;

                Ok(best_index)
            }
        }
    }
    fn arg_max_by(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut slf = slf.try_iter()?;
        match slf.next() {
            None => Err(PyValueError::new_err(
                "Cannot compute `PyoIterator::arg_max_by` of an empty Iterator",
            )),
            Some(first) => {
                let mut best_index = 0;
                let mut best_value = key.call1((first?,))?;

                slf.map(|x| key.call1((x?,)))
                    .enumerate()
                    .try_for_each(|(index, item)| {
                        let value = item?;
                        if value.gt(&best_value)? {
                            best_index = index + 1;
                            best_value = value;
                        }
                        Ok::<(), PyErr>(())
                    })?;

                Ok(best_index)
            }
        }
    }

    fn all_unique(slf: &Bound<'_, Self>) -> PyResult<bool> {
        let slf = slf.try_iter()?;
        let seen = PySet::empty(slf.py())?;
        for item in slf {
            let key_value = item?;
            if seen.contains(&key_value)? {
                return Ok(false);
            }
            seen.add(key_value)?;
        }
        Ok(true)
    }
    fn all_unique_by(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let slf = slf.try_iter()?;
        let seen = PySet::empty(key.py())?;
        for item in slf.map(|item| key.call1((item?,))) {
            let item = item?;
            if seen.contains(&item)? {
                return Ok(false);
            }
            seen.add(item)?;
        }
        Ok(true)
    }

    fn all_equal(slf: &Bound<'_, Self>, key: Option<Bound<'_, PyAny>>) -> PyResult<bool> {
        let slf = slf.try_iter()?;
        let iterator = pylibs::itertools::group_by(slf, key)?;
        for _first in &iterator {
            for _second in iterator {
                return Ok(false);
            }
            return Ok(true);
        }
        Ok(true)
    }
    #[pyo3(signature = (reverse=false, strict=false))]
    fn is_sorted(slf: &Bound<'_, Self>, reverse: bool, strict: bool) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        match slf.next() {
            None => Ok(true),
            Some(first) => {
                let cmp_fn = is_sorted_cmp_fn(strict, reverse);
                let mut prev = first?;
                for item in slf {
                    let curr = item?;
                    if !cmp_fn(&prev, &curr)? {
                        return Ok(false);
                    }
                    prev = curr;
                }
                Ok(true)
            }
        }
    }
    #[pyo3(signature = (key, reverse=false, strict=false))]
    fn is_sorted_by(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        reverse: bool,
        strict: bool,
    ) -> PyResult<bool> {
        let mut iterator = slf.try_iter()?.map(|item| key.call1((item?,)));
        match iterator.next() {
            None => Ok(true),
            Some(first) => {
                let cmp_fn = is_sorted_cmp_fn(strict, reverse);
                let mut prev = first?;
                for item in iterator {
                    let curr = item?;
                    if !cmp_fn(&prev, &curr)? {
                        return Ok(false);
                    }
                    prev = curr;
                }
                Ok(true)
            }
        }
    }

    fn eq(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let py = slf.py();
        let sentinel = pylibs::builtins::sentinel(py)?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    let left = left_res?;
                    let right = right_res?;
                    if left.is(&sentinel) || right.is(&sentinel) || !left.eq(&right)? {
                        return Ok(false);
                    }
                }
                (None, None) => return Ok(true),
                (Some(_), None) | (None, Some(_)) => return Ok(false),
            }
        }
    }
    fn ne(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    if !left_res?.eq(&right_res?)? {
                        return Ok(true);
                    }
                }
                (None, None) => return Ok(false),
                (Some(_), None) | (None, Some(_)) => return Ok(true),
            }
        }
    }
    fn le(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    let left = left_res?;
                    let right = right_res?;
                    if !left.eq(&right)? {
                        return left.lt(&right);
                    }
                }
                (None, None) | (None, Some(_)) => return Ok(true),
                (Some(_), None) => return Ok(false),
            }
        }
    }
    fn lt(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    let left = left_res?;
                    let right = right_res?;
                    if !left.eq(&right)? {
                        return left.lt(&right);
                    }
                }
                (None, None) | (Some(_), None) => return Ok(false),
                (None, Some(_)) => return Ok(true),
            }
        }
    }
    fn gt(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    let left = left_res?;
                    let right = right_res?;
                    if !left.eq(&right)? {
                        return left.gt(&right);
                    }
                }
                (None, None) | (None, Some(_)) => return Ok(false),
                (Some(_), None) => return Ok(true),
            }
        }
    }
    fn ge(slf: &Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let mut slf = slf.try_iter()?;
        let mut other_iterator = other.try_iter()?;
        loop {
            match (slf.next(), other_iterator.next()) {
                (Some(left_res), Some(right_res)) => {
                    let left = left_res?;
                    let right = right_res?;
                    if !left.eq(&right)? {
                        return left.gt(&right);
                    }
                }
                (None, None) | (Some(_), None) => return Ok(true),
                (None, Some(_)) => return Ok(false),
            }
        }
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn for_each(
        slf: &Bound<'_, Self>,
        func: &Bound<'_, PyAny>,
        args: &Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<()> {
        let mut slf = slf.try_iter()?;
        match (args.is_empty(), kwargs) {
            (true, None) => slf.try_for_each(|item| {
                func.call1((&item?,))?;
                Ok(())
            }),
            (true, Some(_)) => slf.try_for_each(|item| {
                func.call((&item?,), kwargs)?;
                Ok(())
            }),
            (false, Some(_)) => slf.try_for_each(|item| {
                func.concat(&item?, args, kwargs)?;
                Ok(())
            }),
            (false, None) => slf.try_for_each(|item| {
                func.concat1(&item?, args)?;
                Ok(())
            }),
        }
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn for_each_star(
        slf: &Bound<'_, Self>,
        func: Bound<'_, PyAny>,
        args: Args<'_>,
        kwargs: Option<&Kwargs<'_>>,
    ) -> PyResult<()> {
        let mut slf = slf.try_iter()?;
        match (args.is_empty(), kwargs) {
            (true, None) => slf.try_for_each(|item| {
                func.call1(item?.cast_exact::<PyTuple>()?)?;
                Ok(())
            }),
            (true, Some(_)) => slf.try_for_each(|item| {
                func.call(item?.cast_exact::<PyTuple>()?, kwargs)?;
                Ok(())
            }),
            (false, None) => slf.try_for_each(|item| {
                func.concat_star1(item?.cast_exact::<PyTuple>()?, &args)?;
                Ok(())
            }),
            (false, Some(_)) => slf.try_for_each(|item| {
                func.concat_star(item?.cast_exact::<PyTuple>()?, &args, kwargs)?;
                Ok(())
            }),
        }
    }
    fn try_for_each(slf: &Bound<'_, Self>, f: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        for item in slf {
            let result = f.call1((&item?,))?;
            match result.cast_exact::<PyoOk>() {
                Ok(_) => (),
                Err(_) => return result.cast_exact::<PyoErr>()?.into_py_any(py),
            }
        }
        PyoOk::new(PyTuple::empty(py).into()).into_py_any(py)
    }

    fn try_find(slf: &Bound<'_, Self>, predicate: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        for item in slf {
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
                        return val
                            .unbind()
                            .pipe(PySome::new)
                            .into_py_any(py)?
                            .pipe(PyoOk::new)
                            .into_py_any(py);
                    }
                }
                Err(_) => {
                    return result
                        .cast_exact::<PyoErr>()?
                        .to_owned()
                        .unbind()
                        .into_any()
                        .pipe(Ok);
                }
            }
        }
        PyNull::get(py)
            .into_py_any(py)
            .map(PyoOk::new)?
            .into_py_any(py)
    }
    fn try_fold(
        slf: &Bound<'_, Self>,
        init: &Bound<'_, PyAny>,
        func: &Bound<'_, PyFunction>,
    ) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        let mut accumulator = init.to_owned().unbind();

        for item in slf {
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

    fn try_reduce(slf: &Bound<'_, Self>, func: &Bound<'_, PyFunction>) -> PyResult<Py<PyAny>> {
        let mut slf = slf.try_iter()?;
        let py = slf.py();
        let first = slf.next();
        match first {
            None => {
                return PyNull::get(py)
                    .into_py_any(py)
                    .map(PyoOk::new)?
                    .into_py_any(py);
            }
            Some(first_val) => {
                let mut accumulator = first_val?.to_owned().unbind();

                for item in slf {
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
                accumulator
                    .pipe(PySome::new)
                    .into_py_any(py)
                    .map(PyoOk::new)?
                    .into_py_any(py)
            }
        }
    }
    fn try_collect<'py>(slf: &Bound<'py, Self>) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        let collected = PyList::empty(py);

        for item in slf {
            let val = item?;
            match val.cast_exact::<PyoOk>() {
                Ok(ok) => collected.append(&ok.get().value)?,
                Err(_) => match val.cast_into_exact::<PySome>() {
                    Ok(some) => collected.append(&some.get().value)?,
                    Err(_) => return PyNull::get_any_ok(py),
                },
            }
        }
        collected
            .pipe_ref(pylibs::pyochain::vec::from_ref)?
            .unbind()
            .pipe(PySome::new)
            .into_py_any(py)
    }

    #[pyo3(signature = (init, func, *args, **kwargs))]
    fn fold_star<'py>(
        slf: &Bound<'py, Self>,
        init: Bound<'py, PyAny>,
        func: Bound<'py, PyAny>,
        args: Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut slf = slf.try_iter()?;
        match (args.is_empty(), kwargs) {
            (true, None) => slf.try_fold(init, |acc, item| {
                func.fold_concat_star1(&acc, item?.cast_exact::<PyTuple>()?, &args)
            }),

            (false, None) => slf.try_fold(init, |acc, item| {
                func.fold_concat_star1(&acc, item?.cast_exact::<PyTuple>()?, &args)
            }),

            (true, Some(_)) => slf.try_fold(init, |acc, item| {
                func.fold_concat_star(&acc, item?.cast_exact::<PyTuple>()?, &args, kwargs)
            }),

            (false, Some(_)) => slf.try_fold(init, |acc, item| {
                func.fold_concat_star(&acc, item?.cast_exact::<PyTuple>()?, &args, kwargs)
            }),
        }
    }
    fn find(slf: &Bound<'_, Self>, predicate: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        slf.filter(|x| {
            predicate
                .call1((x
                    .as_ref()
                    .expect("Error occurred while unwrapping item in `PyoIterator::find`"),))
                .expect("Error occurred while calling predicate function in `PyoIterator::find`")
                .is_truthy()
                .expect("Error occurred while evaluating predicate output in `PyoIterator::find`")
        })
        .next()
        .map(|x| x?.unbind().pipe(PySome::new).into_py_any(py))
        .unwrap_or_else(|| PyNull::get_any_ok(py))
    }
    fn intersperse<'py>(
        slf: &Bound<'py, Self>,
        element: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| tls::Intersperse::new(x, element.unbind()))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn group_by<'py>(
        slf: &Bound<'py, Self>,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::group_by(x, key))
            .map(tls::GroupBy::new)
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_star<'py>(
        slf: Bound<'py, Self>,
        predicate: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterStar::new(x, predicate))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_map<'py>(
        slf: Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterMap::new(x, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_map_star<'py>(
        slf: Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterMapStar::new(x, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map<'py>(slf: Bound<'py, Self>, func: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::map(func, x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map_star<'py>(slf: Bound<'py, Self>, func: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::map_star(func, x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (*funcs))]
    fn map_juxt<'py>(
        slf: Bound<'py, Self>,
        funcs: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::MapJuxt::new(x, funcs))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map_while<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::MapWhile::new(x, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn map_windows<'py>(
        slf: &Bound<'py, Self>,
        length: usize,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        slf.try_iter()
            .and_then(|x| tls::MapWindow::new(x, length))
            .and_then(|x| x.into_bound_py_any(py))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
            .and_then(|x| pylibs::builtins::map(func, x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map_windows_star<'py>(
        slf: &Bound<'py, Self>,
        length: usize,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        slf.try_iter()
            .and_then(|x| tls::MapWindow::new(x, length))
            .and_then(|x| x.into_bound_py_any(py))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
            .and_then(|x| pylibs::itertools::map_star(func, x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn partition<'py>(
        slf: &Bound<'py, Self>,
        predicate: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        let true_list = PyList::empty(py);
        let false_list = PyList::empty(py);
        for item in slf {
            let item = item?;
            if predicate.call1((&item,))?.is_truthy()? {
                true_list.append(item)?;
            } else {
                false_list.append(item)?;
            }
        }
        Ok((
            pylibs::pyochain::vec::from_ref(&true_list)?,
            pylibs::pyochain::vec::from_ref(&false_list)?,
        ))
    }
    fn scan<'py>(
        slf: &Bound<'py, Self>,
        initial: Bound<'py, PyAny>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::Scan::new(x, initial, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    fn unpack_into<'py>(
        slf: &Bound<'py, Self>,
        func: &Bound<'py, PyAny>,
        args: &Args<'py>,
        kwargs: Option<&Kwargs<'py>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let slf = slf.try_iter()?;
        let py = slf.py();
        let unpacked = unsafe {
            Bound::from_owned_ptr(py, ffi::PySequence_Tuple(slf.as_ptr()))
                .cast_into_unchecked::<PyTuple>()
        };
        func.concat_star(&unpacked, args, kwargs)
    }
    fn unique<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(tls::UniqueIdentity::new)
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn unique_by<'py>(slf: Bound<'py, Self>, key: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|iter| tls::UniqueKey::new(iter, key))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn unzip<'py>(slf: Bound<'py, Self>) -> PyResult<(Bound<'py, Self>, Bound<'py, Self>)> {
        slf.try_iter()
            .and_then(|data| pylibs::itertools::tee(data, None))
            .map(|iterators| {
                (
                    tls::Unzip::new(&iterators, 0),
                    tls::Unzip::new(&iterators, 1),
                )
            })
            .map(|(left, right)| {
                let cls = slf.get_type();
                (
                    cls.call1((left,))
                        .map(|x| unsafe { x.cast_into_unchecked::<Self>() }),
                    cls.call1((right,))
                        .map(|x| unsafe { x.cast_into_unchecked::<Self>() }),
                )
            })
            .and_then(|results| match results {
                (Ok(a), Ok(b)) => Ok((a, b)),
                (Err(e), _) | (_, Err(e)) => Err(e),
            })
    }
    fn with_position<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::WithPosition::new(x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (*others))]
    fn zip_longest<'py>(
        slf: Bound<'py, Self>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::zip_longest(x, others))
            .map(tls::ZipLongest::new)
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}

#[inline(always)]
fn is_sorted_cmp_fn(
    strict: bool,
    reverse: bool,
) -> impl Fn(&Bound<'_, PyAny>, &Bound<'_, PyAny>) -> PyResult<bool> {
    match (strict, reverse) {
        (true, false) => |prev: &Bound<'_, PyAny>, curr: &Bound<'_, PyAny>| prev.lt(curr),
        (false, false) => |prev: &Bound<'_, PyAny>, curr: &Bound<'_, PyAny>| prev.le(curr),
        (true, true) => |prev: &Bound<'_, PyAny>, curr: &Bound<'_, PyAny>| prev.gt(curr),
        (false, true) => |prev: &Bound<'_, PyAny>, curr: &Bound<'_, PyAny>| prev.ge(curr),
    }
}
