use crate::collections;
use crate::seq;
use crate::tools;
use pyo3::prelude::*;
use pyochain_macros::py_abc;
#[py_abc(seq::Dict, collections::PyoCounter)]
pub trait ImplPyoReversible {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, tools::Iter>>;
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, tools::Iter>> {
        self.rev(py)
    }
}
