use pyo3::{
    IntoPyObjectExt, PyTypeInfo,
    exceptions::{PyIndexError, PyValueError},
    intern,
    prelude::*,
    types::PySequence,
};
use tap::Pipe;

use crate::{
    abc::PyoCollection,
    iterators,
    option::{PyNull, PySome},
    pyo3_ext::{
        args::{Args, Kwargs},
        pylibs,
    },
    traits::PyoABC,
};
// TODO: check difference once we had `sequence` to pypub struct macro
#[pyclass(module = "pyochain.abc",subclass,  frozen, generic, sequence, extends=PyoCollection)]
pub struct PyoSequence;
#[pymethods]
impl PyoSequence {
    #[pyo3(signature = (*_args, **_kwargs))]
    #[new]
    fn new(_args: &Args<'_>, _kwargs: Option<&Kwargs<'_>>) -> PyClassInitializer<Self> {
        Self::build_init()
    }
    fn __iter__(slf: Bound<'_, Self>) -> iterators::SequenceIterator {
        slf.pipe(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
            .pipe(iterators::SequenceIterator::new)
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
    fn __reversed__(slf: Bound<'_, Self>) -> PyResult<iterators::SequenceReverseIterator> {
        slf.pipe(|x| unsafe { x.cast_into_unchecked::<PySequence>() })
            .pipe(iterators::SequenceReverseIterator::new)
    }

    #[pyo3(signature = (value, start=0, stop=None))]
    fn index(
        slf: Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        start: isize,
        stop: Option<isize>,
    ) -> PyResult<isize> {
        let py = slf.py();
        let length = slf.len()? as isize;

        let mut i = {
            if start < 0 {
                (length + start).max(0)
            } else {
                start
            }
        };
        let stop = stop.map(|s| if s < 0 { s + length } else { s });

        loop {
            if stop.is_some_and(|stop| i >= stop) {
                break;
            } else {
                match slf.get_item(i) {
                    Ok(v) if v.is(value) || v.eq(value)? => return Ok(i),
                    Ok(_) => i += 1,
                    Err(err) if err.is_instance_of::<PyIndexError>(py) => break,
                    Err(err) => return Err(err),
                }
            };
        }

        Err(PyValueError::new_err(""))
    }
    fn count(slf: Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        slf.try_iter()?.try_fold(0, |count, item| {
            item.and_then(|item| {
                if item.is(value) || item.eq(value)? {
                    Ok(count + 1)
                } else {
                    Ok(count)
                }
            })
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
    fn rev<'py>(slf: Bound<'py, Self>) -> PyResult<Bound<'py, iterators::Iter>> {
        slf.as_any()
            .pipe(pylibs::builtins::reversed)
            .pipe(iterators::Iter::new)
    }
}

#[pyclass(module = "pyochain.abc",subclass, frozen, generic, extends=PyoSequence)]
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
    fn extract_if<'py>(
        slf: Bound<'py, Self>,
        predicate: Bound<'py, PyAny>,
        start: usize,
        end: Option<usize>,
    ) -> PyResult<Bound<'py, iterators::Iter>> {
        let py = slf.py();
        unsafe { slf.cast_into_unchecked::<PySequence>() }
            .pipe(|x| iterators::ExtractIf::new(x, predicate, start, end))?
            .into_bound_py_any(py)?
            .try_iter()
            .and_then(iterators::Iter::new)
    }
    #[pyo3(signature = (start=None, end=None))]
    fn drain<'py>(
        slf: Bound<'py, Self>,
        start: Option<usize>,
        end: Option<usize>,
    ) -> PyResult<Bound<'py, iterators::Iter>> {
        let py = slf.py();
        unsafe { slf.cast_into_unchecked::<PySequence>() }
            .pipe(|x| iterators::Drain::new(x, start, end))?
            .into_bound_py_any(py)?
            .try_iter()
            .and_then(iterators::Iter::new)
    }
}
