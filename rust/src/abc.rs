use crate::args::{Args, ConcatWith, Concatenate, Kwargs};
use crate::mixins::Checkable;
use crate::option::{PyNull, PySome};
use crate::result::{PyoErr, PyoOk};
use crate::seq::{SetMut, IntoPyochain, Vec as PyoVec};
use crate::tools as tls;
use crate::{pylibs, tools };
use pyo3::exceptions::{
    PyIndexError, PyKeyError, PyNotImplementedError, PyStopIteration, PyValueError,
};
use pyo3::types::{
    PyBool, PyDict, PyFunction, PyInt, PyIterator, PyList, PyMapping, PyNone, PySequence, PySet,
    PyString, PyTuple, PyType,
};
use pyo3::{BoundObject, IntoPyObjectExt, PyClass, PyTypeInfo, ffi, intern, prelude::*};
use tap::prelude::*;


pub trait PyoABC: PyTypeInfo + PyClass {
    fn build_init() -> PyClassInitializer<Self>;
}

impl PyoABC for PyoIterable {
    
    fn build_init() -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable).add_subclass(Self)
    }
}
impl PyoABC for PyoIterator {
    fn build_init() -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }

}
impl PyoABC for PyoCollection {
    fn build_init() -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }}
impl PyoABC for PyoSequence {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoSet {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableSet {
    fn build_init() -> PyClassInitializer<Self> {
        PyoSet::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableSequence {
    fn build_init() -> PyClassInitializer<Self> {
        PyoSequence::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMapping {
    fn build_init() -> PyClassInitializer<Self> {
        PyoCollection::build_init().add_subclass(Self)
    }
}
impl PyoABC for PyoMutableMapping {
    fn build_init() -> PyClassInitializer<Self> {
        PyoMapping::build_init().add_subclass(Self)
    }
}
#[pyclass(subclass, frozen, generic, extends=Checkable)]
pub struct PyoIterable;

#[pymethods]
impl PyoIterable {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
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
        Self::build_init()
    }
    fn __iter__<'py>(slf: Bound<'py, Self>) -> Bound<'py, Self> {
        slf
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
    #[pyo3(signature = (key=None))]
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
        collected.into_pyochain()?
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
    ) -> PyResult<(Bound<'py, PyoVec>, Bound<'py, PyoVec>)> {
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
            true_list.into_pyochain()?,
            false_list.into_pyochain()?
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
    fn sort<'py>(slf: &Bound<'py, Self>, reverse: bool) -> PyResult<Bound<'py, PyoVec>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::sorted(&x, reverse)?.into_pyochain())
    }
    #[pyo3(signature = (key, *,reverse=false))]
    fn sort_by<'py>(
        slf: &Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        reverse: bool,
    ) -> PyResult<Bound<'py, PyoVec>> {
        slf.try_iter()
            .and_then(|x| pylibs::builtins::sorted_by(&x, reverse, key)?.into_pyochain())
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
        PyClassInitializer::from(Checkable).add_subclass(Self)
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
        PyClassInitializer::from(Checkable).add_subclass(Self)
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

#[pyclass(subclass, frozen, generic, extends=PyoSized)]
pub struct PyoMappingView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}

#[pymethods]
impl PyoMappingView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyClassInitializer::from(Checkable)
            .add_subclass(PyoSized)
            .add_subclass(Self {
                _mapping: mapping.unbind(),
            })
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoCollection;

#[pymethods]
impl PyoCollection {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
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
#[pyclass(subclass, frozen, generic, extends=PyoIterable)]
pub struct PyoReversible;

#[pymethods]
impl PyoReversible {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        PyoIterable::build_init().add_subclass(Self)
    }
    /// We use unsafe code here because calling `reversed` with `PyOnceLock` pattern is 2x slower than pure python for some reason.
    fn rev(slf: Bound<'_, Self>) -> PyResult<Py<tls::Iter>> {
        slf.as_any()
            .pipe(pylibs::builtins::reversed)
            .into_any()
            .pipe(|x| tls::Iter::new(x))
    }
}

// TODO: check difference once we had `sequence` to pypub struct macro
#[pyclass(subclass,  frozen, generic, sequence, extends=PyoCollection)]
pub struct PyoSequence;
#[pymethods]
impl PyoSequence {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn __iter__(slf: Bound<'_, Self>) -> tools::SequenceIterator {
        slf.pipe(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
            .pipe(tools::SequenceIterator::new)
    }
    fn __contains__(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
        for v in slf.try_iter()? {
            let item = v?;
            if item.is(value) || item.eq(value)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn __reversed__(slf: Bound<'_, Self>) -> PyResult<tools::SequenceReverseIterator> {
        slf.pipe(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
            .pipe(tools::SequenceReverseIterator::new)
    }

    #[pyo3(signature = (value, start=None, stop=None))]
    fn index(
        slf: Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        start: Option<usize>,
        stop: Option<usize>,
    ) -> PyResult<usize> {
        let py = slf.py();
        let start = start.map(|x| slf.len().map(|len| len + x)).transpose()?;
        let stop = stop.map(|x| slf.len().map(|len| len + x)).transpose()?;

        let mut i = start.unwrap_or_default();
        while stop.map(|x| i < x).unwrap_or(true) {
            let v = slf.get_item(i).map_err(|x| {
                if x.is_instance_of::<PyIndexError>(py) {
                    PyValueError::new_err("")
                } else {
                    x
                }
            })?;
            if v.is(value) || v.eq(value)? {
                break;
            } else {
                i += 1;
                continue;
            }
        }
        Ok(i)
    }
    fn count(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        slf.try_iter().map(|iterator| {
            iterator
                .map(|x| x.expect("Unexpected error while iterating over sequence"))
                .filter(|x| {
                    x.is(value)
                        || x.eq(value)
                            .expect("Unexpected error while comparing values")
                })
                .count()
        })
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
            .into_any()
            .pipe(|x| tls::Iter::new(x))
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSequence)]
pub struct PyoMutableSequence;
#[pymethods]
impl PyoMutableSequence {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }


    fn __iadd__(slf: Bound<'_, Self>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "extend"), (values,))?;
        Ok(())
    }

    fn append(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "insert"), (slf.len()?, value))?;
        Ok(())
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        loop {
            match slf.del_item(0) {
                Ok(_) => continue,
                Err(err) => {
                    if err.is_instance_of::<PyIndexError>(slf.py()) {
                        return Ok(());
                    } else {
                        return Err(err);
                    }
                }
            }
        }
    }

    fn extend(slf: Bound<'_, Self>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if values.is(&slf) {
            unsafe { slf.cast_into_unchecked::<PySequence>() }.in_place_concat(
                unsafe { values.cast_unchecked::<PySequence>() }
                    .to_list()?
                    .as_sequence(),
            )?;
        } else {
            for v in values.try_iter()? {
                slf.call_method1(intern!(py, "append"), (v?,))?;
            }
        };
        Ok(())
    }

    #[pyo3(signature = (index=-1))]
    fn pop(slf: Bound<'_, Self>, index: isize) -> PyResult<Bound<'_, PyAny>> {
        let v = slf.get_item(index)?;
        slf.del_item(index)?;
        Ok(v)
    }

    fn remove(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.del_item(unsafe { slf.cast_unchecked::<PySequence>() }.index(value)?)
    }
    fn reverse(slf: Bound<'_, Self>) -> PyResult<()> {
        let n = slf.len()?;
        for i in 0..n / 2 {
            let tmp = slf.get_item(i)?;
            slf.set_item(i, slf.get_item(n - i - 1)?)?;
            slf.set_item(n - i - 1, tmp)?;
        }
        Ok(())
    }
    fn retain(slf: Bound<'_, Self>, predicate: &Bound<'_, PyAny>) -> PyResult<()> {
        let seq = unsafe { slf.cast_into_unchecked::<PySequence>() };
        let mut write_idx = 0;
        //TODO: why TF do we create an Iterator from a range instead of the Sequence itself??
        for val in (0..seq.len()?).map(|x| seq.get_item(x)) {
            let curr = val?;
            if predicate.call1((&curr,))?.is_truthy()? {
                seq.set_item(write_idx, curr)?;
                write_idx += 1;
            }
        }
        seq.del_slice(write_idx, usize::MAX)?;
        Ok(())
    }

    fn truncate(slf: Bound<'_, Self>, length: usize) -> PyResult<()> {
        unsafe { slf.cast_into_unchecked::<PySequence>() }.del_slice(length, usize::MAX)
    }
    #[pyo3(signature = (predicate, start=0, end=None))]
    fn extract_if(
        slf: Bound<'_, Self>,
        predicate: Bound<'_, PyAny>,
        start: usize,
        end: Option<usize>,
    ) -> PyResult<Py<tls::Iter>> {
        let py = slf.py();
        unsafe { slf.cast_into_unchecked::<PySequence>() }
            .pipe(|x| tls::ExtractIf::new(x, predicate, start, end))?
            .into_bound_py_any(py)
            .and_then(tls::Iter::new)
    }
    #[pyo3(signature = (start=None, end=None))]
    fn drain(
        slf: Bound<'_, Self>,
        start: Option<usize>,
        end: Option<usize>,
    ) -> PyResult<Py<tls::Iter>> {
        let py = slf.py();
        unsafe { slf.cast_into_unchecked::<PySequence>() }
            .pipe(|x| tls::Drain::new(x, start, end))?
            .into_bound_py_any(py)
            .and_then(tls::Iter::new)
    }
}
#[pyclass(subclass, frozen, generic, extends=PyoCollection)]
pub struct PyoSet;
#[pymethods]
impl PyoSet {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    #[classmethod]
    fn _from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call1((it,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
            .map_err(|e| {
                let name = cls.name().unwrap();
                let msg = format!("hint: As a `PyoSet` subclass, `{}::__init__` must accept a single `Iterable` argument. If you override it, make sure to override `PyoSet::_from_iterable` as well.", name);
                e.add_note(cls.py(), msg,).unwrap(); 
                e})
    
}
    
    #[inline]
    #[classmethod]
    fn _py_from_iterable<'py>(
        cls: &Bound<'py, PyType>,
        it: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        cls.call_method1(intern!(cls.py(), "_from_iterable"), (it,))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }

    fn __and__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            return Err(PyNotImplementedError::new_err(""));
        }
        slf.try_iter()?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if other.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&slf.get_type(), &x))
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn __or__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            return Err(PyNotImplementedError::new_err(""));
        }
        slf.try_iter()?
            .chain(other.try_iter()?)
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                init.append(item)?;
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&slf.get_type(), &x))
    }

    fn __sub__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            return Err(PyNotImplementedError::new_err(""));
        }
        other
            .try_iter()
            .map_err(|_| PyNotImplementedError::new_err(""))
            .and_then(|iterator| {
                let cls = slf.get_type();
                let other_set = Self::_py_from_iterable(&cls, iterator.as_any())?;
                slf.try_iter()?
                    .try_fold(PyList::empty(py), |init, x| {
                        let item = x?;
                        if !other_set.contains(&item)? {
                            init.append(item)?;
                        }
                        Ok::<_, PyErr>(init)
                    })?
                    .into_bound_py_any(py)
                    .and_then(|x| Self::_py_from_iterable(&cls, &x))
            })
    }

    fn __rsub__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        let cls = slf.get_type();
        other
            .try_iter()
            .map_err(|_| PyNotImplementedError::new_err(""))
            .and_then(|x| {
                if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
                    Self::_py_from_iterable(&cls, x.as_any())?.try_iter()
                } else {
                    Ok(x)
                }
            })?
            .try_fold(PyList::empty(py), |init, x| {
                let item = x?;
                if !slf.contains(&item)? {
                    init.append(item)?;
                }
                Ok::<_, PyErr>(init)
            })?
            .into_bound_py_any(py)
            .and_then(|x| Self::_py_from_iterable(&cls, &x))
    }

    fn __xor__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        let py = slf.py();
        if other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            other
                .try_iter()
                .map_err(|_| PyNotImplementedError::new_err(""))
                .and_then(|iterator| Self::_py_from_iterable(&slf.get_type(), iterator.as_any()))
                .and_then(|x| (slf.sub(&x))?.bitor((x).sub(slf)?))
                .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }

    fn __rand__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__and__(slf, other)
    }
    fn __ror__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        Self::__or__(slf, other)
    }
    fn __rxor__<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        Self::__xor__(slf, other)
    }
    fn __eq__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        if other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            Ok(slf.len()? == other.len()? && slf.le(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }
    fn __le__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            return Err(PyNotImplementedError::new_err(""));
        }
        if slf.len()? > other.len()? {
            return Ok(false);
        }
        for elem in slf.try_iter()? {
            if !other.contains(elem?)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    fn __ge__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        if !other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            return Err(PyNotImplementedError::new_err(""));
        }
        if slf.len()? < other.len()? {
            return Ok(false);
        }
        for elem in other.try_iter()? {
            if !slf.contains(elem?)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    fn __lt__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        if other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            Ok(slf.len()? < other.len()? && slf.le(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }

    fn __gt__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        if other.is_instance(&pylibs::collections::abc::Set(py)?)? {
            Ok(slf.len()? > other.len()? && slf.ge(other)?)
        } else {
            Err(PyNotImplementedError::new_err(""))
        }
    }

    fn isdisjoint(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        for value in other.try_iter()? {
            if slf.contains(value?)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn _hash(slf: Bound<'_, Self>) -> PyResult<isize> {
        let max = isize::MAX / 2;
        let mask = 2 * max + 1;
        let n = slf.len()? as isize;
        let mut h = 1927868237 * (n + 1);
        h &= mask;
        for x in slf.try_iter()? {
            let hx = x?.hash()?;
            h ^= (hx ^ (hx << 16) ^ 89869747) * 3644798167;
            h &= mask;
        }
        h ^= (h >> 11) ^ (h >> 25);
        h = h * 69069 + 907133923;
        h &= mask;
        if h > max {
            h -= mask + 1;
        }
        if h == -1 {
            h = 590923713;
        }
        Ok(h)
    }

    fn is_subset(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.le(other)
    }
    fn is_subset_strict(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.lt(other)
    }
    fn eq(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.eq(other)
    }
    fn is_superset(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.ge(other)
    }
    fn is_superset_strict(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.gt(other)
    }
    fn is_disjoint(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        Self::isdisjoint(slf, other)
    }
    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.bitand(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, Self>> {
        slf.bitor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.sub(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, Self>> {
        slf.bitxor(other)
            .map(|x| unsafe { x.cast_into_unchecked::<Self>() })
    }
}
#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoValuesView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoValuesView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init()
            .add_subclass(Self {
                _mapping: mapping.unbind(),
            })
    }

    fn __contains__(&self, value: Bound<'_, PyAny>) -> PyResult<bool> {
        let mapping = self._mapping.bind(value.py());
        for item in mapping.try_iter()?.map(|key| mapping.get_item(&key?)) {
            let v = item?;
            if v.is(&value) || v.eq(&value)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<tls::ValuesViewIterator> {
        let py = slf.py();
        slf.get()
            ._mapping
            .clone_ref(py)
            .into_bound(py)
            .pipe(tls::ValuesViewIterator::new)
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoKeysView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoKeysView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init()
            .add_subclass(Self {
                _mapping: mapping.unbind(),
            })
    }

    #[classmethod]
    fn _from_iterable<'py>(
        cls: Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySet>> {
        PySet::type_object(cls.py()).call1((it,)).map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
    }

    fn __contains__(&self, key: Bound<'_, PyAny>) -> PyResult<bool> {
        self._mapping.bind(key.py()).contains(key)
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        slf.get()._mapping.bind(slf.py()).try_iter()
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoItemsView {
    #[pyo3(get)]
    pub _mapping: Py<PyAny>,
}
#[pymethods]
impl PyoItemsView {
    #[new]
    fn new(mapping: Bound<'_, PyAny>) -> PyClassInitializer<Self> {
        PyoSet::build_init()
            .add_subclass(Self {
                _mapping: mapping.unbind(),
            })
    }

    #[classmethod]
    fn _from_iterable<'py>(
        cls: Bound<'py, PyType>,
        it: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PySet>> {
        PySet::type_object(cls.py()).call1((it,)).map(|x| unsafe { x.cast_into_unchecked::<PySet>() })
    }

    fn __contains__(&self, item: (Bound<'_, PyAny>, Bound<'_, PyAny>)) -> PyResult<bool> {
        let (key, value) = item;
        let py = key.py();

        let v = self
            ._mapping
            .bind(py)
            .get_item(key)
            .and_then(|v| Ok(v.is(&value) || v.eq(&value)?));
        match v {
            Ok(v) => Ok(v),
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(py) {
                    Ok(false)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn __iter__(slf: Bound<'_, Self>) -> PyResult<tls::ItemsViewIterator> {
        let py = slf.py();
        slf.get()
            ._mapping
            .clone_ref(py)
            .into_bound(py)
            .pipe(tls::ItemsViewIterator::new)
    }

    fn intersection<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitand(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn union<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyAny>) -> PyResult<Bound<'py, SetMut>> {
        slf.bitor(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.sub(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }

    fn symmetric_difference<'py>(
        slf: Bound<'py, Self>,
        other: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, SetMut>> {
        slf.bitxor(other).and_then(|x| unsafe { x.cast_into_unchecked::<PySet>() }.into_pyochain())
    }
}

#[pyclass(subclass, frozen, generic, extends=PyoSet)]
pub struct PyoMutableSet;
#[pymethods]
impl PyoMutableSet {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn _py_add(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "add"), (value,))?;
        Ok(())
    }
    fn _py_discard(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        slf.call_method1(intern!(slf.py(), "discard"), (value,))?;
        Ok(())
    }

    fn __ior__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        it.try_iter()?
            .try_for_each(|value| Self::_py_add(&slf, &value?))
    }

    fn __iand__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        slf.sub(&it)?
            .try_iter()?
            .try_for_each(|value| Self::_py_discard(&slf, &value?))
    }

    fn __isub__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            for value in it.try_iter()? {
                Self::_py_discard(&slf, &value?)?;
            }
        }
        Ok(())
    }

    fn __ixor__<'py>(slf: Bound<'py, Self>, it: Bound<'py, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let cls = slf.get_type();
        if it.is(&slf) {
            slf.call_method0(intern!(py, "clear"))?;
        } else {
            let pyset = if !it.is_instance(&pylibs::collections::abc::Set(py)?)? {
                PyoSet::_py_from_iterable(&cls, &it)?.into_any()
            } else {
                it
            };
            for value in pyset.try_iter()? {
                let v = value?;
                if slf.contains(&v)? {
                    Self::_py_discard(&slf, &v)?;
                } else {
                    Self::_py_add(&slf, &v)?;
                }
            }
        }

        Ok(())
    }


    fn remove(slf: Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        if !slf.contains(&value)? {
            Err(PyKeyError::new_err(format!("{}", value)))
        } else {
            Self::_py_discard(&slf, &value)?;
            Ok(())
        }
    }

    fn pop(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyAny>> {
        match slf.try_iter()?.next() {
            None => Err(PyKeyError::new_err("")),
            Some(value) => value.and_then(|x| {
                Self::_py_discard(&slf, &x)?;
                Ok(x)
            }),
        }
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        slf.try_iter()?
            .try_for_each(|x| Self::_py_discard(&slf, &x?))
    }
}
#[pyclass(subclass, frozen, generic, mapping, extends=PyoCollection)]
pub struct PyoMapping;
#[pymethods]
impl PyoMapping {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn __contains__(slf: Bound<'_, Self>, key: Bound<'_, PyAny>) -> PyResult<bool> {
        slf.get_item(key).map(|_| true).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(slf.py()) {
                Ok(false)
            } else {
                Err(err)
            }
        })
    }

    fn __eq__(slf: Bound<'_, Self>, other: Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        other
            .cast::<PyMapping>()
            .map_err(|_| PyNotImplementedError::new_err(""))
            .and_then(|other| {
                slf.call_method0(intern!(py, "items"))?
                    .pipe_ref(PyDict::from_sequence)?
                    .eq(other
                        .call_method0(intern!(py, "items"))?
                        .pipe_ref(PyDict::from_sequence)?)
            })
    }

    fn keys(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoKeysView>> {
        PyoKeysView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoKeysView>() })
    }

    fn values(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoValuesView>> {
        PyoValuesView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoValuesView>() })
    }

    fn items(slf: Bound<'_, Self>) -> PyResult<Bound<'_, PyoItemsView>> {
        PyoItemsView::type_object(slf.py())
            .call1((slf,))
            .map(|x| unsafe { x.cast_into_unchecked::<PyoItemsView>() })
    }

    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(py) {
                Ok(default.unwrap_or_else(|| PyNone::get(py).into_bound_py_any(py).unwrap()))
            } else {
                Err(err)
            }
        })
    }

    fn get_item<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key)
            .and_then(|value| PySome::new(value.unbind()).into_bound_py_any(py))
            .or_else(|err| {
                if err.is_instance_of::<PyKeyError>(py) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            })
    }
}

#[pyclass(subclass, frozen, generic, mapping, extends=PyoMapping)]
pub struct PyoMutableMapping;

#[pymethods]
impl PyoMutableMapping {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    #[pyo3(signature = (key, default=None))]
    fn pop<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match slf.get_item(key) {
            Ok(value) => {
                slf.del_item(key)?;
                Ok(value)
            }
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    default.ok_or(err)
                } else {
                    Err(err)
                }
            }
        }
    }

    fn popitem(slf: Bound<'_, Self>) -> PyResult<(Bound<'_, PyAny>, Bound<'_, PyAny>)> {
        slf.try_iter()?
            .next()
            .map(|k| {
                let key = k?;
                let value = slf.get_item(&key)?;
                slf.del_item(&key)?;
                Ok((key, value))
            })
            .unwrap_or_else(|| Err(PyKeyError::new_err("")))
    }

    fn clear(slf: Bound<'_, Self>) -> PyResult<()> {
        let py = slf.py();
        loop {
            match slf.call_method0(intern!(py, "popitem")) {
                Ok(_) => continue,
                Err(err) => {
                    if err.is_instance_of::<PyKeyError>(py) {
                        return Ok(());
                    } else {
                        return Err(err);
                    }
                }
            }
        }
    }

    #[pyo3(signature = (other=None, **kwds))]
    fn update(
        slf: Bound<'_, Self>,
        other: Option<Bound<'_, PyAny>>,
        kwds: Option<Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        other.map(|x| {
            if x.is_instance_of::<PyMapping>() {
                x.try_iter()?
                    .try_for_each(|key| key.and_then(|k| slf.set_item(&k, x.get_item(&k)?)))
            } else if x.hasattr("keys")? {
                x.call_method0(intern!(slf.py(), "keys"))
                    .unwrap()
                    .try_iter()?
                    .try_for_each(|key| key.and_then(|k| slf.set_item(&k, x.get_item(&k)?)))
            } else {
                x.try_iter()?.try_for_each(|item| {
                    let tup = item?.cast_into::<PyTuple>()?;
                    let (key, value) = (tup.get_item(0)?, tup.get_item(1)?);
                    slf.set_item(&key, &value)
                })
            }
        });
        kwds.map(|kwds| {
            kwds.items()
                .iter()
                .map(|x| unsafe { x.cast_into_unchecked::<PyTuple>() })
                .map(|x| unsafe { (x.get_item_unchecked(0), x.get_item_unchecked(1)) })
                .try_for_each(|(key, value)| slf.set_item(&key, &value))
        });
        Ok(())
    }
    #[pyo3(signature = (key, default=None))]
    fn setdefault<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key).or_else(|err| {
            if err.is_instance_of::<PyKeyError>(py) {
                let default = default
                    .map(Ok)
                    .unwrap_or_else(|| PyNone::get(py).into_bound_py_any(py))?;
                slf.set_item(key, &default)?;
                Ok(default)
            } else {
                Err(err)
            }
        })
    }

    fn insert<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        value: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let previous = slf.get_item(key);
        slf.set_item(&key, &value)?;
        previous
            .map(|x| PySome::new(x.unbind()).into_bound_py_any(py))
            .unwrap_or_else(|_| PyNull::get(py).into_bound_py_any(py))
    }

    fn try_insert<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
        value: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        if slf.contains(&key)? {
            PyoErr::new(
                PyKeyError::new_err(format!(
                    "Key {} already exists with value {}.",
                    key,
                    slf.get_item(&key)?
                ))
                .into_py_any(py)?,
            )
            .into_bound_py_any(py)
        } else {
            slf.set_item(&key, &value)?;
            value.unbind().pipe(PyoOk::new).into_bound_py_any(py)
        }
    }

    fn remove<'py>(slf: Bound<'py, Self>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        slf.get_item(key)
            .and_then(|value| {
                slf.del_item(key)?;
                value.unbind().pipe(PySome::new).into_bound_py_any(py)
            })
            .or_else(|err| {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            })
    }

    fn remove_entry<'py>(
        slf: Bound<'py, Self>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        match slf.get_item(key) {
            Ok(value) => {
                slf.del_item(key)?;
                PyTuple::new(py, [key, &value])?
                    .into_any()
                    .unbind()
                    .pipe(PySome::new)
                    .into_bound_py_any(py)
            }
            Err(err) => {
                if err.is_instance_of::<PyKeyError>(slf.py()) {
                    PyNull::get(py).into_bound_py_any(py)
                } else {
                    Err(err)
                }
            }
        }
    }
}
