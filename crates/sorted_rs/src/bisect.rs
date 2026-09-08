/// Module for bisect functions, adapted from the Python standard library's bisect module.\
use pyo3::prelude::*;
pub(super) trait Bisect {
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> PyResult<usize>;
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> PyResult<usize>;
}
impl Bisect for [Py<PyAny>] {
    #[inline]
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = item.py();
        resolve(self.len(), |mid| item.lt(self[mid].bind(py)))
    }
    #[inline]
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> PyResult<usize> {
        let py = item.py();
        resolve(self.len(), |mid| Ok(!self[mid].bind(py).lt(item)?))
    }
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
