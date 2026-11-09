use crate::abc;
use derive_more::{Deref, From};
use pyo3::prelude::*;
use sorted_rs::iter::{self, PySortedIter};

macro_rules! impl_sorted_iter {
    ($($iter:ty => [$pyiter:ident, $pyiterrev:ty]),+ $(,)?) => {
        $(
            #[derive(From, Deref)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $pyiter($iter);
            impl iter::PySortedIter for $pyiter {
                type Iter = $iter;
                type IterRev = $pyiterrev;
            }
            #[pymethods]
            impl $pyiter {
                fn __next__<'py>(slf: &Bound<'py, Self>) -> Option<Bound<'py, PyAny>> {
                    iter::PySortedIter::py_next(slf)
                }
                fn __length_hint__(&self) -> usize {
                      self.0.len()
                }
                fn rev<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, <Self as PySortedIter>::IterRev>> {
                    iter::PySortedIter::rev(slf)
                }
            }
        )+
    };
}

impl_sorted_iter!(
    iter::IterFwd => [SortedIter, SortedIterRev],
    iter::IterRev => [SortedIterRev, SortedIter],
    iter::IterBounded => [SortedIterBounded, SortedIterBoundedRev],
    iter::IterBoundedRev => [SortedIterBoundedRev, SortedIterBounded],
);
