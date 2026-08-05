use pyo3::prelude::*;
use std::cmp::Ordering;

///TODO: We should find a way to avoid `expect` here, whilst keeping in-place sorting behavior.
const MSG: &str = "Unexpected error during comparison";
#[inline(always)]
pub fn py_cmp(py: Python<'_>, a: &Py<PyAny>, b: &Py<PyAny>) -> Ordering {
    a.bind(py).lt(b.bind(py)).map(gt_if_true).expect(MSG)
}
#[inline(always)]
pub fn py_cmp_by_key(a: &Py<PyAny>, b: &Py<PyAny>, key: &Bound<'_, PyAny>) -> Ordering {
    key.call1((a,))
        .and_then(|key_a| key_a.lt(key.call1((b,))?))
        .map(gt_if_true)
        .expect(MSG)
}
#[inline(always)]
fn gt_if_true(b: bool) -> Ordering {
    if b { Ordering::Greater } else { Ordering::Less }
}
