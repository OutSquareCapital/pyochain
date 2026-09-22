//! Python `builtins` functions and objects
use crate::args::{ArgsConcat, CallConcat};
use pyo3::{
    call::PyCallArgs,
    ffi, intern,
    prelude::*,
    sync::PyOnceLock,
    types::{PyBool, PyDict, PyIterator, PyList, PyTuple},
};
use tap::prelude::*;

const BUILTINS: &str = "builtins";
static OBJECT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ALL: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ANY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static MAX: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static MIN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SUM: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ENUMERATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static MAP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static FILTER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static SORTED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ZIP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
/// Create a unique sentinel object. Equivalent to `object()` in Python. On >=3.15, this will become unneded thanks to the new sentinel builtin.
#[inline(always)]
pub fn sentinel(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    OBJECT.import(py, BUILTINS, "object")?.call0()
}
#[inline(always)]
pub fn all<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyBool>> {
    ALL.import(iterator.py(), BUILTINS, "all")?
        .call1((iterator,))
        .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
}
#[inline(always)]
pub fn any<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyBool>> {
    ANY.import(iterator.py(), BUILTINS, "any")?
        .call1((iterator,))
        .map(|x| unsafe { x.cast_into_unchecked::<PyBool>() })
}

pub fn enumerate<'py>(
    iterator: &Bound<'py, PyIterator>,
    start: usize,
) -> PyResult<Bound<'py, PyIterator>> {
    ENUMERATE
        .import(iterator.py(), BUILTINS, "enumerate")?
        .call1((iterator, start))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}

#[inline(always)]
pub fn filter<'py>(
    func: Option<&Bound<'py, PyAny>>,
    iterator: &Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    FILTER
        .import(iterator.py(), BUILTINS, "filter")?
        .call1((func, iterator))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn max<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyAny>> {
    MAX.import(iterator.py(), BUILTINS, "max")?
        .call1((iterator,))
}
#[inline(always)]
pub fn max_of<'py, A: PyCallArgs<'py>>(py: Python<'py>, objs: A) -> PyResult<Bound<'py, PyAny>> {
    MAX.import(py, BUILTINS, "max")?.call1(objs)
}
#[inline(always)]
pub fn max_by<'py>(
    iterator: &Bound<'py, PyIterator>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(iterator.py());
    kwargs.set_item(intern!(iterator.py(), "key"), key)?;
    MAX.import(iterator.py(), BUILTINS, "max")?
        .call((iterator,), Some(&kwargs))
}
#[inline(always)]
pub fn min<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyAny>> {
    MIN.import(iterator.py(), BUILTINS, "min")?
        .call1((iterator,))
}
#[inline(always)]
pub fn min_of<'py, A: PyCallArgs<'py>>(py: Python<'py>, objs: A) -> PyResult<Bound<'py, PyAny>> {
    MIN.import(py, BUILTINS, "min")?.call1(objs)
}
#[inline(always)]
pub fn min_by<'py>(
    iterator: &Bound<'py, PyIterator>,
    key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(iterator.py());
    kwargs.set_item(intern!(iterator.py(), "key"), key)?;
    MIN.import(iterator.py(), BUILTINS, "min")?
        .call((iterator,), Some(&kwargs))
}
#[inline(always)]
pub fn sum<'py>(iterator: &Bound<'py, PyIterator>, start: &i32) -> PyResult<Bound<'py, PyAny>> {
    SUM.import(iterator.py(), BUILTINS, "sum")?
        .call1((iterator, start))
}
#[inline(always)]
pub fn map<'py>(
    func: &Bound<'py, PyAny>,
    iterator: &Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    MAP.import(iterator.py(), BUILTINS, "map")?
        .call1((func, iterator))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}
/// first arg is a function, the rest is a variable number of iterables.
#[inline(always)]
pub fn map_with<'py, A: ArgsConcat<'py>>(
    py: Python<'py>,
    args: A,
) -> PyResult<Bound<'py, PyIterator>> {
    MAP.import(py, BUILTINS, "map")?
        .call_concat1(args)
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
#[must_use]
pub fn reversed<'py>(sequence: &Bound<'py, PyAny>) -> Bound<'py, PyIterator> {
    unsafe {
        ffi::PyObject_CallOneArg(
            (&raw const ffi::PyReversed_Type)
                .cast::<ffi::PyObject>()
                .cast_mut(),
            sequence.as_ptr(),
        )
        .pipe(|x| Bound::from_owned_ptr(sequence.py(), x))
        .pipe(|x| x.cast_into_unchecked::<PyIterator>())
    }
}
#[inline(always)]
pub fn sorted<'py>(
    iterator: &Bound<'py, PyIterator>,
    reverse: bool,
) -> PyResult<Bound<'py, PyList>> {
    let py = iterator.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "reverse"), reverse)?;
    SORTED
        .import(py, BUILTINS, "sorted")?
        .call((iterator,), Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}
#[inline(always)]
pub fn sorted_by<'py>(
    iterator: &Bound<'py, PyIterator>,
    reverse: bool,
    key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let py = iterator.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "reverse"), reverse)?;
    kwargs.set_item(intern!(py, "key"), key)?;
    SORTED
        .import(py, BUILTINS, "sorted")?
        .call((iterator,), Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}
#[inline(always)]
pub fn zip<'py>(
    iterator: &Bound<'py, PyIterator>,
    others: &Bound<'py, PyTuple>,
    strict: bool,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "strict"), strict)?;
    ZIP.import(py, BUILTINS, "zip")?
        .call_concat((iterator.as_any(), others), Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}
