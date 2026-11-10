use pyo3::{basic::CompareOp, prelude::*};
use pyo3_ext::prelude::PyAnyExtMethods;
use std::ops::Not;

/// Output of a bisect operation.\
/// May fail if the `__lt__` dunder can raise an exception.
type BisectRes = PyResult<usize>;

/// bisect function, adapted from the Python standard library's bisect module.
#[inline(always)]
pub fn bisect<const LEFT: bool>(list: &[Py<PyAny>], item: &Bound<'_, PyAny>) -> BisectRes {
    if LEFT {
        resolve(list, item, |item, mid| {
            mid.rich_compare_bool(item, CompareOp::Lt).map(Not::not)
        })
    } else {
        resolve(list, item, |item, mid| {
            item.rich_compare_bool(mid, CompareOp::Lt)
        })
    }
}

#[inline(always)]
fn resolve<F>(vec: &[Py<PyAny>], item: &Bound<'_, PyAny>, mut func: F) -> BisectRes
where
    F: FnMut(&Bound<'_, PyAny>, &Bound<'_, PyAny>) -> PyResult<bool>,
{
    let py = item.py();
    let mut high = vec.len();
    let mut low = 0;
    while low < high {
        let mid = low.midpoint(high);
        if func(item, vec[mid].bind(py))? {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    Ok(low)
}
