use crate::{collections, dict, iterators};
use pyo3::prelude::*;
use pyochain_macros::py_abc;
#[py_abc(dict::Dict, collections::PyoCounter)]
pub trait ImplPyoReversible {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>>;
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, iterators::Iter>> {
        self.rev(py)
    }
}
