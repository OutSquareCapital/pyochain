use pyo3::{
    call::PyCallArgs,
    intern,
    prelude::*,
    types::{PyDict, PyIterator, PyList},
};

use crate::pylibs::registery::heapq;
#[inline(always)]
pub fn merge<'py, A: PyCallArgs<'py>>(
    py: Python<'py>,
    iterables: A,
    key: Option<Bound<'py, PyAny>>,
    reverse: bool,
) -> PyResult<Bound<'py, PyIterator>> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(intern!(py, "key"), key)?;
    kwargs.set_item(intern!(py, "reverse"), reverse)?;

    heapq::MERGE
        .import(py)
        .call(iterables, Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}

#[inline(always)]
pub fn nlargest<'py>(
    n: isize,
    iterable: &Bound<'py, PyAny>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    heapq::NLARGEST
        .import(iterable.py())
        .call1((n, iterable, key))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}

#[inline(always)]
pub fn nsmallest<'py>(
    n: isize,
    iterable: &Bound<'py, PyAny>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    heapq::NSMALLEST
        .import(iterable.py())
        .call1((n, iterable, key))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}
#[inline(always)]
pub fn heapify_max(heap: &Bound<'_, PyList>) -> PyResult<()> {
    heapq::HEAPIFY_MAX
        .import(heap.py())
        .call1((heap,))
        .map(|_| ())
}

pub fn heapify(heap: &Bound<'_, PyList>) -> PyResult<()> {
    heapq::HEAPIFY.import(heap.py()).call1((heap,)).map(|_| ())
}
pub fn heappop<'py>(heap: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
    heapq::HEAPPOP.import(heap.py()).call1((heap,))
}
pub fn heappush(heap: &Bound<'_, PyList>, item: Bound<'_, PyAny>) -> PyResult<()> {
    heapq::HEAPPUSH
        .import(heap.py())
        .call1((heap, item))
        .map(|_| ())
}
pub fn heappushpop<'py>(
    heap: &Bound<'py, PyList>,
    item: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    heapq::HEAPPUSHPOP.import(heap.py()).call1((heap, item))
}

pub fn heapreplace<'py>(
    heap: &Bound<'py, PyList>,
    item: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    heapq::HEAPREPLACE.import(heap.py()).call1((heap, item))
}
