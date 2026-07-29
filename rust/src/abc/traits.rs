use crate::{collections, dict, tools};
use pyo3::prelude::*;
use pyochain_macros::py_abc;
#[py_abc(dict::Dict, collections::PyoCounter)]
pub trait ImplPyoReversible {
    fn rev<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, tools::Iter>>;
    fn __reversed__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, tools::Iter>> {
        self.rev(py)
    }
}
