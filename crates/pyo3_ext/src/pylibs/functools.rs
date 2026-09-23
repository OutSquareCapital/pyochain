//! Python `functools` module functions and objects
use pyo3::{prelude::*, sync::PyOnceLock, types::PyIterator};

use crate::tuple;
const FUNCTOOLS: &str = "functools";
static REDUCE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
#[inline(always)]
pub fn reduce<'py>(
    function: &Bound<'py, PyAny>,
    iterable: &Bound<'py, PyIterator>,
    initial: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = function.py();
    let args = match initial {
        Some(initial) => tuple!(function, iterable, initial)?,
        None => tuple!(function, iterable)?,
    };
    REDUCE.import(py, FUNCTOOLS, "reduce")?.call1(args)
}
