use pyo3::{basic::CompareOp, prelude::*};
use pyo3_ext::prelude::PyAnyExtMethods;
use std::ops::Not;

/// Output of a bisect operation.\
/// May fail if the `__lt__` dunder can raise an exception.
type BisectRes = PyResult<usize>;

/// Trait for bisect functions, adapted from the Python standard library's bisect module.
pub trait Bisect {
    fn bisect<const LEFT: bool>(&self, item: &Bound<'_, PyAny>) -> BisectRes;
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> BisectRes;
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> BisectRes;
}
impl Bisect for [Py<PyAny>] {
    #[inline(always)]
    fn bisect<const LEFT: bool>(&self, item: &Bound<'_, PyAny>) -> BisectRes {
        if LEFT {
            resolve(self, item, |item, mid| {
                mid.rich_compare_bool(item, CompareOp::Lt).map(Not::not)
            })
        } else {
            resolve(self, item, |item, mid| {
                item.rich_compare_bool(mid, CompareOp::Lt)
            })
        }
    }
    #[inline(always)]
    fn bisect_left(&self, item: &Bound<'_, PyAny>) -> BisectRes {
        self.bisect::<true>(item)
    }
    #[inline(always)]
    fn bisect_right(&self, item: &Bound<'_, PyAny>) -> BisectRes {
        self.bisect::<false>(item)
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
