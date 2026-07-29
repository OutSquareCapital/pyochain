/// Module for bisect functions, adapted from the Python standard library's bisect module.\
/// Adapted to only handle `pyochain::PyoVec` for both simplicity and performance.
use pyo3::{prelude::*, types::PyList};
/// The following documentation and code is adapted from the Python standard library's bisect module.
///Return the index where to insert item x in list a, assuming a is sorted.

///The return value i is such that all e in a[:i] have e <= x, and all e in
///a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
///insert just after the rightmost x already there.

///Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.
#[inline]
pub(super) fn right(lst: &Bound<'_, PyList>, item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let mut high = lst.len();
    let mut lo = 0;
    // Note, the comparison uses "<" to match the
    // __lt__() logic in list.sort() and in heapq.
    while lo < high {
        let mid = (lo + high) / 2;
        if item.lt(lst.get_item(mid)?)? {
            high = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}
/// The following documentation and code is adapted from the Python standard library's bisect module.\
/// Return the index where to insert item x in list a, assuming a is sorted.\
/// The return value i is such that all e in a[:i] have e < x, and all e in a[i:] have e >= x.\
/// So if x already appears in the list, a.insert(i, x) will insert just before the leftmost x already there.\
/// Optional args lo (default 0) and hi (default len(a)) bound the slice of a to be searched.\
#[inline]
pub(super) fn left(lst: &Bound<'_, PyList>, item: &Bound<'_, PyAny>) -> PyResult<usize> {
    let mut hi = lst.len();
    let mut lo = 0;
    // Note, the comparison uses "<" to match the
    // __lt__() logic in list.sort() and in heapq.
    while lo < hi {
        let mid = (lo + hi) / 2;
        if lst.get_item(mid)?.lt(item)? {
            lo = mid + 1;
        } else {
            hi = mid
        }
    }
    Ok(lo)
}
