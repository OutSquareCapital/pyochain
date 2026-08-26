/// Module for bisect functions, adapted from the Python standard library's bisect module.\
/// Adapted to only handle `pyochain::PyoVec` for both simplicity and performance.
use pyo3::prelude::*;
/// The following documentation and code is adapted from the Python standard library's bisect module.
///Return the index where to insert item x in list a, assuming a is sorted.\
///The return value i is such that all e in a[:i] have e <= x, and all e in
///a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
///insert just after the rightmost x already there.
///Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.
#[inline]
pub(super) fn right(lst: &[Py<PyAny>], item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let py = item.py();
    resolve(lst.len(), |mid| item.lt(lst[mid].bind(py)))
}
#[inline]
pub(super) fn left(lst: &[Py<PyAny>], item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let py = item.py();
    resolve(lst.len(), |mid| Ok(!lst[mid].bind(py).lt(item)?))
}

#[inline(always)]
fn resolve(mut high: usize, mut func: impl FnMut(usize) -> PyResult<bool>) -> PyResult<usize> {
    let mut low = 0;
    while low < high {
        let mid = low.midpoint(high);
        if func(mid)? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}
