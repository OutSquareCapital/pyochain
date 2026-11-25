use std::cmp;

use pyo3::prelude::*;
use pyo3_ext::prelude::PyAnyExtMethods;
pub trait Item: Sized {
    fn key<'py>(&self) -> &Bound<'py, PyAny>;

    ///Generic, safe comparison.
    /// For fast, *unsafe* comparisons, see [`SortKey`](crate::sort::SortKey).
    #[inline(always)]
    fn cmp(&self, b: &Self) -> PyResult<cmp::Ordering> {
        self.key().lt_fast(b.key())
    }
}

impl Item for Py<PyAny> {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        let py = unsafe { Python::assume_attached() };
        self.bind(py)
    }
}
impl Item for (Py<PyAny>, Py<PyAny>) {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        let py = unsafe { Python::assume_attached() };
        self.0.bind(py)
    }
}
impl Item for &Py<PyAny> {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        (**self).key()
    }
}
impl Item for (&Py<PyAny>, &Py<PyAny>) {
    #[inline(always)]
    fn key<'py>(&self) -> &Bound<'py, PyAny> {
        self.0.key()
    }
}

pub trait ItemRef: Item + Copy {
    type Owned: Item;
    fn owned(&self, py: Python<'_>) -> Self::Owned;
}
impl ItemRef for &Py<PyAny> {
    type Owned = Py<PyAny>;
    fn owned(&self, py: Python<'_>) -> Self::Owned {
        self.clone_ref(py)
    }
}
impl ItemRef for (&Py<PyAny>, &Py<PyAny>) {
    type Owned = (Py<PyAny>, Py<PyAny>);
    fn owned(&self, py: Python<'_>) -> Self::Owned {
        (self.0.clone_ref(py), self.1.clone_ref(py))
    }
}
