use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyIterator, PyTuple};
use tap::prelude::*;
const ITERTOOLS: &str = "itertools";
static TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

#[inline(always)]
pub fn tee<'py>(
    iterator: Bound<'py, PyIterator>,
    n: Option<usize>,
) -> PyResult<Bound<'py, PyTuple>> {
    TEE.import(iterator.py(), ITERTOOLS, "tee")?
        .call1((iterator, n.unwrap_or(2)))?
        .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyTuple>() })
        .pipe(Ok)
}
#[inline(always)]
pub fn group_by<'py>(
    iterator: Bound<'py, PyIterator>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyIterator>> {
    GROUP_BY
        .import(iterator.py(), ITERTOOLS, "groupby")?
        .call1((iterator, key))?
        .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        .pipe(Ok)
}
