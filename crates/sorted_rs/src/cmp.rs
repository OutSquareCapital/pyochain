use pyo3::{basic::CompareOp, prelude::*};
use pyo3_ext::prelude::PyAnyExtMethods;
use std::cmp::Ordering;

//TODO: We should find a way to avoid `expect` here, whilst keeping in-place sorting behavior.
const MSG: &str = "Unexpected error during comparison";
#[inline(always)]
pub fn py_cmp(py: Python<'_>, a: &Py<PyAny>, b: &Py<PyAny>) -> Ordering {
    lt_if_true(a.bind(py), b.bind(py)).expect(MSG)
}
#[inline(always)]
pub fn py_cmp_by_key(a: &Py<PyAny>, b: &Py<PyAny>, key: &Bound<'_, PyAny>) -> Ordering {
    key.call1((a,))
        .and_then(|key_a| lt_if_true(&key_a, &key.call1((b,))?))
        .expect(MSG)
}
#[inline(always)]
fn lt_if_true(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Ordering> {
    a.rich_compare_bool(b, CompareOp::Lt).map(|lt| {
        if lt {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
}
