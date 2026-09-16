//! Python `itertools` module functions and objects

use pyo3::{
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{PyDict, PyInt, PyIterator, PyNone, PyTuple},
};
use tap::Pipe;

use crate::args::{ArgsConcat, CallConcat};

const ITERTOOLS: &str = "itertools";
static ACCUMULATE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static BATCHED: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static TEE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static GROUP_BY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ZIP_LONGEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static FILTER_FALSE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static MAP_STAR: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COUNT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static TAKE_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static DROP_WHILE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COMBINATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COMBINATIONS_WITH_REPLACEMENT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static COMPRESS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static CYCLE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static PAIRWISE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static PRODUCT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static PERMUTATIONS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static ISLICE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// `itertools::chain` class.
pub mod chain {

    use pyo3::{intern, types::PyIterator};

    use crate::args::{ArgsConcat, CallConcat};

    use super::*;
    static CHAIN: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

    #[inline(always)]
    pub fn new<'py, A: ArgsConcat<'py>>(
        py: Python<'py>,
        iterables: A,
    ) -> PyResult<Bound<'py, PyIterator>> {
        CHAIN
            .import(py, ITERTOOLS, "chain")?
            .call_concat1(iterables)
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }

    #[inline(always)]
    pub fn from_iterable<'py>(
        iterable: &Bound<'py, PyIterator>,
    ) -> PyResult<Bound<'py, PyIterator>> {
        let py = iterable.py();
        CHAIN
            .import(py, ITERTOOLS, "chain")?
            .getattr(intern!(py, "from_iterable"))?
            .call1((iterable,))
            .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
    }
}
#[inline(always)]
pub fn count<'py>(py: Python<'py>, start: &i32, step: &i32) -> PyResult<Bound<'py, PyIterator>> {
    COUNT
        .import(py, ITERTOOLS, "count")?
        .call1((start, step))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}

#[inline(always)]
pub fn tee(iterator: Bound<'_, PyIterator>, n: usize) -> PyResult<Bound<'_, PyTuple>> {
    TEE.import(iterator.py(), ITERTOOLS, "tee")?
        .call1((iterator, n))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyTuple>() })
}
#[inline(always)]
pub fn take_while<'py>(
    predicate: &Bound<'py, PyAny>,
    iterator: &Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    TAKE_WHILE
        .import(iterator.py(), ITERTOOLS, "takewhile")?
        .call1((predicate, iterator))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn drop_while<'py>(
    predicate: &Bound<'py, PyAny>,
    iterator: &Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    DROP_WHILE
        .import(iterator.py(), ITERTOOLS, "dropwhile")?
        .call1((predicate, iterator))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn accumulate<'py>(
    iterator: &Bound<'py, PyIterator>,
    func: Option<Bound<'py, PyAny>>,
    initial: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "initial"), initial)?;
    ACCUMULATE
        .import(py, ITERTOOLS, "accumulate")?
        .call((iterator, func), Some(&kwargs))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn batched<'py>(
    iterator: &Bound<'py, PyIterator>,
    n: &Bound<'py, PyInt>,
    strict: &bool,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "strict"), strict)?;
    BATCHED
        .import(py, ITERTOOLS, "batched")?
        .call((iterator, n), Some(&kwargs))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn compress<'py>(
    iterator: &Bound<'py, PyIterator>,
    selectors: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyIterator>> {
    COMPRESS
        .import(iterator.py(), ITERTOOLS, "compress")?
        .call1((iterator, selectors))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn cycle<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyIterator>> {
    CYCLE
        .import(iterator.py(), ITERTOOLS, "cycle")?
        .call1((iterator,))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn permutations<'py>(
    iterator: &Bound<'py, PyIterator>,
    r: Option<usize>,
) -> PyResult<Bound<'py, PyIterator>> {
    PERMUTATIONS
        .import(iterator.py(), ITERTOOLS, "permutations")?
        .call1((iterator, r))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn combinations<'py>(
    iterator: &Bound<'py, PyIterator>,
    r: &Bound<'py, PyInt>,
) -> PyResult<Bound<'py, PyIterator>> {
    COMBINATIONS
        .import(iterator.py(), ITERTOOLS, "combinations")?
        .call1((iterator, r))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn combinations_with_replacement<'py>(
    iterator: &Bound<'py, PyIterator>,
    r: &Bound<'py, PyInt>,
) -> PyResult<Bound<'py, PyIterator>> {
    COMBINATIONS_WITH_REPLACEMENT
        .import(iterator.py(), ITERTOOLS, "combinations_with_replacement")?
        .call1((iterator, r))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn filter_false<'py>(
    func: Option<Bound<'py, PyAny>>,
    iterator: &Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    FILTER_FALSE
        .import(iterator.py(), ITERTOOLS, "filterfalse")?
        .call1((func, iterator))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn group_by<'py>(
    iterator: &Bound<'py, PyIterator>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyIterator>> {
    GROUP_BY
        .import(iterator.py(), ITERTOOLS, "groupby")?
        .call1((iterator, key))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn pairwise<'py>(iterator: &Bound<'py, PyIterator>) -> PyResult<Bound<'py, PyIterator>> {
    PAIRWISE
        .import(iterator.py(), ITERTOOLS, "pairwise")?
        .call1((iterator,))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn product<'py, A: ArgsConcat<'py>>(
    py: Python<'py>,
    iterables: A,
    repeat: usize,
) -> PyResult<Bound<'py, PyIterator>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "repeat"), repeat)?;
    PRODUCT
        .import(py, ITERTOOLS, "product")?
        .call_concat(iterables, Some(&kwargs))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn map_star<'py>(
    func: Bound<'py, PyAny>,
    iterable: Bound<'py, PyIterator>,
) -> PyResult<Bound<'py, PyIterator>> {
    MAP_STAR
        .import(iterable.py(), ITERTOOLS, "starmap")?
        .call1((func, iterable))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn nth<'py>(
    iterator: &Bound<'py, PyIterator>,
    n: usize,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = iterator.py();
    ISLICE
        .import(py, ITERTOOLS, "islice")?
        .call1((iterator, n, n + 1))?
        .pipe(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
        .next()
        .transpose()
}
#[inline(always)]
pub fn slice<'py>(
    iterator: &Bound<'py, PyIterator>,
    start: Option<&Bound<'py, PyInt>>,
    stop: Option<&Bound<'py, PyInt>>,
    step: Option<&Bound<'py, PyInt>>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    ISLICE
        .import(py, ITERTOOLS, "islice")?
        .call1((iterator, start, stop, step))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn skip<'py>(
    iterator: &Bound<'py, PyIterator>,
    n: &Bound<'py, PyInt>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    ISLICE
        .import(py, ITERTOOLS, "islice")?
        .call1((iterator, n, PyNone::get(py)))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}

#[inline(always)]
pub fn zip_longest<'py>(
    iterator: &Bound<'py, PyIterator>,
    others: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    ZIP_LONGEST
        .import(py, ITERTOOLS, "zip_longest")?
        .call_concat1((iterator.as_any(), others))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn step_by<'py>(
    iterator: &Bound<'py, PyIterator>,
    step: &Bound<'py, PyInt>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    ISLICE
        .import(py, ITERTOOLS, "islice")?
        .call1((iterator, 0, PyNone::get(py), step))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
#[inline(always)]
pub fn take<'py>(
    iterator: &Bound<'py, PyIterator>,
    stop: &Bound<'py, PyInt>,
) -> PyResult<Bound<'py, PyIterator>> {
    let py = iterator.py();
    ISLICE
        .import(py, ITERTOOLS, "islice")?
        .call1((iterator, stop))
        .map(|obj| unsafe { obj.cast_into_unchecked::<PyIterator>() })
}
