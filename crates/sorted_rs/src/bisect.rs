use pyo3::prelude::*;
use std::ops::Not;

/// Trait for bisect functions, adapted from the Python standard library's bisect module.
pub trait Bisect {
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> PyResult<usize>;
}
impl Bisect for [Py<PyAny>] {
    #[inline(always)]
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        resolve(self, item, |item, mid| mid.lt(item).map(Not::not))
    }
    #[inline(always)]
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        resolve(self, item, |item, mid| item.lt(mid))
    }
}

#[inline(always)]
fn resolve(
    vec: &[Py<PyAny>],
    item: &Bound<'_, PyAny>,
    mut func: impl FnMut(&Bound<'_, PyAny>, &Bound<'_, PyAny>) -> PyResult<bool>,
) -> PyResult<usize> {
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
