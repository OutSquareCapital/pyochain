/// Module for bisect functions, adapted from the Python standard library's bisect module.\
use pyo3::prelude::*;
#[inline]
pub fn right(lst: &[Py<PyAny>], item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let py = item.py();
    resolve(lst.len(), |mid| item.lt(lst[mid].bind(py)))
}
#[inline]
pub fn left(lst: &[Py<PyAny>], item: &Bound<'_, PyAny>) -> PyResult<usize> {
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
