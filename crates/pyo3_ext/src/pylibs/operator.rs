use pyo3::prelude::*;

use crate::pylibs::registery::ITEMGETTER;

#[inline(always)]
pub fn itemgetter(py: Python<'_>, index: isize) -> PyResult<Bound<'_, PyAny>> {
    ITEMGETTER.import(py).call1((index,))
}
