use pyo3::{
    call::PyCallArgs,
    intern,
    prelude::*,
    sync::PyOnceLock,
    types::{PyDict, PyIterator, PyList},
};
const HEAPQ: &str = "heapq";
static MERGE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NLARGEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static NSMALLEST: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPIFY_MAX: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPIFY: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPPOP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPPUSH: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPPUSHPOP: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
static HEAPREPLACE: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
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

    MERGE
        .import(py, HEAPQ, "merge")?
        .call(iterables, Some(&kwargs))
        .map(|x| unsafe { x.cast_into_unchecked::<PyIterator>() })
}

#[inline(always)]
pub fn nlargest<'py>(
    n: isize,
    iterable: &Bound<'py, PyAny>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    NLARGEST
        .import(iterable.py(), HEAPQ, "nlargest")?
        .call1((n, iterable, key))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}

#[inline(always)]
pub fn nsmallest<'py>(
    n: isize,
    iterable: &Bound<'py, PyAny>,
    key: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    NSMALLEST
        .import(iterable.py(), HEAPQ, "nsmallest")?
        .call1((n, iterable, key))
        .map(|x| unsafe { x.cast_into_unchecked::<PyList>() })
}
#[inline(always)]
pub fn heapify_max(heap: &Bound<'_, PyList>) -> PyResult<()> {
    HEAPIFY_MAX
        .import(heap.py(), HEAPQ, "_heapify_max")?
        .call1((heap,))
        .map(|_| ())
}

pub fn heapify(heap: &Bound<'_, PyList>) -> PyResult<()> {
    HEAPIFY
        .import(heap.py(), HEAPQ, "heapify")?
        .call1((heap,))
        .map(|_| ())
}
pub fn heappop<'py>(heap: &Bound<'py, PyList>) -> PyResult<Bound<'py, PyAny>> {
    HEAPPOP.import(heap.py(), HEAPQ, "heappop")?.call1((heap,))
}
pub fn heappush(heap: &Bound<'_, PyList>, item: Bound<'_, PyAny>) -> PyResult<()> {
    HEAPPUSH
        .import(heap.py(), HEAPQ, "heappush")?
        .call1((heap, item))
        .map(|_| ())
}
pub fn heappushpop<'py>(
    heap: &Bound<'py, PyList>,
    item: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    HEAPPUSHPOP
        .import(heap.py(), HEAPQ, "heappushpop")?
        .call1((heap, item))
}
pub fn heapreplace<'py>(
    heap: &Bound<'py, PyList>,
    item: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    HEAPREPLACE
        .import(heap.py(), HEAPQ, "heapreplace")?
        .call1((heap, item))
}
