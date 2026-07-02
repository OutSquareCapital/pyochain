use crate::args::{Args, ConcatWith, Concatenate, Kwargs};
use crate::mixins::Checkable;
use crate::option::{PyNull, PySome};
use crate::pylibs;
use crate::result::{PyoErr, PyoOk};
use crate::tools as tls;
use pyo3::exceptions::{PyIndexError, PyNotImplementedError, PyStopIteration, PyValueError};
use pyo3::types::{
    PyBool, PyFunction, PyInt, PyIterator, PyList, PySequence, PySet, PyString, PyTuple, PyType,
};
use pyo3::{BoundObject, IntoPyObjectExt, PyTypeInfo, ffi, intern, prelude::*};
use tap::prelude::*;

#[pymodule(name = "_iterator")]
pub fn abc(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyoIterable>()?;
    m.add_class::<PyoIterator>()?;
    m.add_class::<PyoContainer>()?;
    m.add_class::<PyoSized>()?;
    m.add_class::<PyoCollection>()?;
    m.add_class::<PyoReversible>()?;
    m.add_class::<PyoSequence>()?;
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
    fn __iter__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        not_impl_error(slf.as_any(), "PyoIterable", "__iter__")
    }
    fn iter<'py>(slf: Bound<'py, Self>) -> PyResult<Py<tls::Iter>> {
        slf.into_any().pipe(tls::Iter::new)
    }
}
#[pyclass(subclass, frozen, generic, extends=PyoIterable)]
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
    fn __iter__<'py>(slf: Bound<'py, Self>) -> Bound<'py, Self> {
        slf
    }
    fn __next__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        not_impl_error(slf.as_any(), "PyoIterator", "__next__")
    }
    #[classmethod]
    fn _from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        iterable: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((iterable,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[classmethod]
    pub fn once<'py>(
        cls: &Bound<'py, PyType>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((PyTuple::new(cls.py(), &[value])?,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (func, *args, **kwargs))]
    #[classmethod]
    fn once_with<'py>(
        cls: &Bound<'py, PyType>,
        func: Bound<'py, PyAny>,
        args: Args<'_>,
        kwargs: Option<Kwargs<'_>>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((tls::OnceWith::new(func, args, kwargs),))
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
    #[pyo3(signature = (obj, n=None))]
    #[classmethod]
    fn repeat<'py>(
        cls: &Bound<'py, PyType>,
        obj: &Bound<'py, PyAny>,
        n: Option<&Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((pylibs::itertools::repeat(obj, n)?,))
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
            None => pylibs::builtins::any(&slf),
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
            None => pylibs::builtins::all(&slf),
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
        let iterator = pylibs::itertools::group_by(&slf, key)?;
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
    #[pyo3(signature = (*others))]
    fn chain<'py>(slf: Bound<'py, Self>, others: &Args<'py>) -> PyResult<Bound<'py, Self>> {
        let cls = slf.get_type();

        slf.into_any()
            .concat_with(others)
            .and_then(|x| pylibs::itertools::chain::new(&x))
            .and_then(|x| cls.call1((&x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (n = 0))]
    fn enumerate<'py>(slf: &Bound<'py, Self>, n: usize) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::enumerate(&x, n))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
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
            .as_any()
            .pipe_ref(pylibs::pyochain::vec::new)?
            .unbind()
            .into_any()
            .pipe(PySome::new)
            .into_py_any(py)
    }
    fn collect<'py>(
        slf: &Bound<'py, Self>,
        collector: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter().and_then(|x| collector.call1((x,)))
    }
    fn collect_into<'py>(
        slf: &Bound<'py, Self>,
        collector: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySequence>> {
        slf.try_iter()
            .and_then(|x| collector.call_method1(intern!(slf.py(), "extend"), (x,)))?;
        collector
            .pipe(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
            .pipe(Ok)
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

    fn skip_while<'py>(
        slf: &Bound<'py, Self>,
        predicate: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::drop_while(predicate, &x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn take_while<'py>(
        slf: &Bound<'py, Self>,
        predicate: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::take_while(predicate, &x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (func=None, initial=None))]
    fn accumulate<'py>(
        slf: &Bound<'py, Self>,
        func: Option<Bound<'py, PyAny>>,
        initial: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::accumulate(&x, func, initial))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (n, strict=false))]
    fn batched<'py>(
        slf: &Bound<'py, Self>,
        n: &Bound<'py, PyInt>,
        strict: bool,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::batched(&x, n, &strict))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    #[pyo3(signature = (*selectors))]
    fn compress<'py>(
        slf: &Bound<'py, Self>,
        selectors: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::compress(&x, selectors))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn cycle<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::cycle(&x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn combinations<'py>(
        slf: &Bound<'py, Self>,
        r: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::combinations(&x, r))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn combinations_with_replacement<'py>(
        slf: &Bound<'py, Self>,
        r: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::combinations_with_replacement(&x, r))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn fold<'py>(
        slf: &Bound<'py, Self>,
        init: &Bound<'py, PyAny>,
        func: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter()
            .and_then(|x| pylibs::functools::reduce(func, &x, Some(init)))
    }
    fn group_by<'py>(
        slf: &Bound<'py, Self>,
        key: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::group_by(&x, key))
            .map(tls::GroupBy::new)
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn join<'py>(
        slf: &Bound<'py, Self>,
        sep: &Bound<'py, PyString>,
    ) -> PyResult<Bound<'py, PyString>> {
        slf.try_iter()
            .and_then(|x| sep.call_method1(intern!(sep.py(), "join"), (x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<PyString>() })
    }
    fn pairwise<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::pairwise(&x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (r=None))]
    fn permutations<'py>(slf: &Bound<'py, Self>, r: Option<usize>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::permutations(&x, r))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (func=None))]
    fn filter<'py>(
        slf: &Bound<'py, Self>,
        func: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::filter(func, &x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_star<'py>(
        slf: &Bound<'py, Self>,
        predicate: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterStar::new(x, predicate))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (func=None))]
    fn filter_false<'py>(
        slf: &Bound<'py, Self>,
        func: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::filter_false(func, &x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_map<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterMap::new(x, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn filter_map_star<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .map(|x| tls::FilterMapStar::new(x, func))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn find_map<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let mut iter = slf.try_iter()?.map(|x| func.call1((x?,)));
        loop {
            match iter.next() {
                None => return PyNull::get(py).into_bound(py).into_any().pipe(Ok),
                Some(result) => {
                    let item = result?;
                    match item.is(PyNull::get(py)) {
                        false => return Ok(item),
                        true => continue,
                    }
                }
            }
        }
    }
    fn flat_map<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::map(&func, &x))
            .and_then(|x| pylibs::itertools::chain::from_iterable(&x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn flatten<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::chain::from_iterable(&x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map<'py>(slf: &Bound<'py, Self>, func: &Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::map(func, &x))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn map_star<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
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
        func: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        slf.try_iter()
            .and_then(|x| tls::MapWindow::new(x, length))
            .and_then(|x| x.into_bound_py_any(py))
            .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
            .and_then(|x| pylibs::builtins::map(func, &x))
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
    #[pyo3(signature = (func, *iterables))]
    fn map_with<'py>(
        slf: &Bound<'py, Self>,
        func: Bound<'py, PyAny>,
        iterables: &Args<'py>,
    ) -> PyResult<Bound<'py, Self>> {
        let cls = slf.get_type();
        func.concat_with_2(slf.try_iter()?.as_any(), iterables)
            .pipe_ref(pylibs::builtins::map_with)
            .and_then(|x| cls.call1((&x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn max<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter().and_then(|x| pylibs::builtins::max(&x))
    }
    fn max_by<'py>(slf: &Bound<'py, Self>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::max_by(&x, key))
    }
    fn min<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter().and_then(|x| pylibs::builtins::min(&x))
    }
    fn min_by<'py>(slf: &Bound<'py, Self>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::min_by(&x, key))
    }
    fn nth<'py>(slf: &Bound<'py, Self>, n: usize) -> PyResult<Py<PyAny>> {
        let py = slf.py();
        slf.try_iter()
            .and_then(|x| pylibs::itertools::nth(&x, n))
            .and_then(|x| {
                x.map(|y| y.unbind().pipe(PySome::new).into_py_any(py))
                    .unwrap_or_else(|| PyNull::get(py).into_py_any(py))
            })
    }
    fn next<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.try_iter()?
            .next()
            .map(|x| x?.unbind().pipe(PySome::new).into_bound_py_any(py))
            .unwrap_or_else(|| PyNull::get(py).into_bound_py_any(py))
    }
    fn peekable<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, tls::Peekable>> {
        let py = slf.py();
        slf.try_iter()
            .and_then(tls::Peekable::new)
            .map(|x| x.into_bound(py))
    }
    fn partition<'py>(
        slf: &Bound<'py, Self>,
        predicate: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PySequence>, Bound<'py, PySequence>)> {
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
            pylibs::pyochain::vec::new(&true_list)?,
            pylibs::pyochain::vec::new(&false_list)?,
        ))
    }
    #[pyo3(signature = (*others, repeat=1))]
    fn product<'py>(
        slf: Bound<'py, Self>,
        others: &Args<'py>,
        repeat: usize,
    ) -> PyResult<Bound<'py, Self>> {
        let cls = slf.get_type();
        slf.into_any()
            .concat_with(others)
            .and_then(|x| pylibs::itertools::product(&x, repeat))
            .and_then(|x| cls.call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn reduce<'py>(
        slf: &Bound<'py, Self>,
        func: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter()
            .and_then(|x| pylibs::functools::reduce(func, &x, None))
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
    #[pyo3(signature = (start=None, stop=None, step=None))]
    fn slice<'py>(
        slf: &Bound<'py, Self>,
        start: Option<&Bound<'py, PyInt>>,
        stop: Option<&Bound<'py, PyInt>>,
        step: Option<&Bound<'py, PyInt>>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::slice(&x, &start, &stop, &step))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn skip<'py>(slf: &Bound<'py, Self>, n: &Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::skip(&x, n))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (*, reverse=false))]
    fn sort<'py>(slf: &Bound<'py, Self>, reverse: bool) -> PyResult<Bound<'py, PySequence>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::sorted(&x, reverse))
            .and_then(|x| pylibs::pyochain::vec::new(&x))
    }
    #[pyo3(signature = (key, *,reverse=false))]
    fn sort_by<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PySequence>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::sorted_by(&x, reverse, key))
            .and_then(|x| pylibs::pyochain::vec::new(&x))
    }
    fn step_by<'py>(
        slf: &Bound<'py, Self>,
        step: &Bound<'py, PyInt>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::step_by(&x, step))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3 (signature = (start=0))]
    fn sum<'py>(slf: &Bound<'py, Self>, start: i32) -> PyResult<Bound<'py, PyAny>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::sum(&x, &start))
    }
    fn tail<'py>(slf: &Bound<'py, Self>, n: usize) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| tls::Tail::new(x, n))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn take<'py>(slf: &Bound<'py, Self>, n: &Bound<'py, PyInt>) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::take(&x, n))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (n=2))]
    fn tee<'py>(slf: &Bound<'py, Self>, n: usize) -> PyResult<Bound<'py, PyTuple>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::tee(x, n))?
            .iter()
            .map(|x| slf.get_type().call1((x,)))
            .collect::<PyResult<Vec<_>>>()
            .and_then(|v| PyTuple::new(slf.py(), v))
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
            .and_then(|data| pylibs::itertools::tee(data, 2))
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
    #[pyo3(signature = (*others, strict=false))]
    fn zip<'py>(
        slf: Bound<'py, Self>,
        others: &Bound<'py, PyTuple>,
        strict: bool,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::zip(&x, others, strict))
            .and_then(|x| slf.get_type().call1((x,)))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    #[pyo3(signature = (*others))]
    fn zip_longest<'py>(
        slf: Bound<'py, Self>,
        others: &Bound<'py, PyTuple>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.try_iter()
            .and_then(|x| pylibs::itertools::zip_longest(&x, others))
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
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoContainer;

#[pymethods]
impl PyoContainer {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self {})
    }
    fn __contains__<'py>(slf: Bound<'py, Self>, _other: &Bound<'py, PyAny>) -> PyResult<bool> {
        not_impl_error(slf.as_any(), "PyoContainer", "__contains__")
    }
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
}

#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoSized;

#[pymethods]
impl PyoSized {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self {})
    }
    fn __len__<'py>(slf: Bound<'py, Self>) -> PyResult<usize> {
        not_impl_error(slf.as_any(), "PyoSized", "__len__")
    }
    #[pyo3(name = "len")]
    fn pyo_len(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoCollection;

#[pymethods]
impl PyoCollection {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoIterable)
            .add_subclass(Self {})
    }
    fn __contains__<'py>(slf: Bound<'py, Self>, _other: &Bound<'py, PyAny>) -> PyResult<bool> {
        not_impl_error(slf.as_any(), "PyoContainer", "__contains__")
    }
    fn __len__<'py>(slf: Bound<'py, Self>) -> PyResult<usize> {
        not_impl_error(slf.as_any(), "PyoSized", "__len__")
    }
    #[pyo3(name = "contains")]
    fn pyo_contains(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        slf.contains(value)
    }
    #[pyo3(name = "len")]
    fn pyo_len(slf: Bound<'_, Self>) -> PyResult<usize> {
        slf.len()
    }
    #[pyo3(name = "is_empty")]
    fn pyo_is_empty(slf: Bound<'_, Self>) -> PyResult<bool> {
        slf.is_empty()
    }
}
#[pyclass(subclass, frozen, generic)]
pub struct PyoReversible;

#[pymethods]
impl PyoReversible {
    fn __reversed__<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, PyIterator>> {
        not_impl_error(slf.as_any(), "PyoReversible", "__reversed__")
    }
    /// We use unsafe code here because calling `reversed` with `PyOnceLock` pattern is 2x slower than pure python for some reason.
    fn rev(slf: Bound<'_, Self>) -> PyResult<Py<tls::Iter>> {
        slf.as_any()
            .pipe(pylibs::builtins::reversed)
            .pipe(|x| tls::Iter::new(x))
    }
}

// TODO: check difference once we had `sequence` to pyclass macro
#[pyclass(subclass, frozen, generic, extends=PyoCollection)]
pub struct PyoSequence;
#[pymethods]
impl PyoSequence {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoIterable)
            .add_subclass(PyoCollection)
            .add_subclass(Self {})
    }
    fn __getitem__<'py>(
        slf: &Bound<'py, Self>,
        _index: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        not_impl_error(slf.as_any(), "PyoSequence", "__getitem__")
    }
    fn first<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.get_item(0)
    }

    fn last<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.get_item(-1)
    }

    fn get<'py>(slf: Bound<'py, Self>, index: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let res = slf.get_item(index);
        match res {
            Ok(ok) => ok.unbind().pipe(PySome::new).into_bound_py_any(py),
            Err(err) => {
                if err.matches(py, PyIndexError::type_object(py)).unwrap() {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            }
        }
    }
    fn rev(slf: Bound<'_, Self>) -> PyResult<Py<tls::Iter>> {
        slf.as_any()
            .pipe(pylibs::builtins::reversed)
            .pipe(|x| tls::Iter::new(x))
    }
}

#[inline]
fn not_impl_error<'py, T>(cls: &Bound<'py, PyAny>, parent: &str, method: &str) -> PyResult<T> {
    let name = cls.get_type().name()?.to_str()?.to_owned();
    let txt = format!(
        "As a subclass of '{}', '{}' must be implemented by {}",
        parent, method, name
    );
    Err(PyNotImplementedError::new_err(txt))
}
