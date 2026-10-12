use crate::abc;
use derive_more::{Deref, From};
use pyo3::prelude::*;
use sorted_rs::iter;

macro_rules! impl_sorted_iter {
    ($($iter:ty => $pyiter:ident),+ $(,)?) => {
        $(
            #[derive(From, Deref)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $pyiter($iter);
            impl iter::PySortedIter<$iter> for $pyiter {}
            #[pymethods]
            impl $pyiter {
                fn __next__(slf: Bound<'_, Self>) -> Option<Bound<'_, PyAny>> {
                    iter::PySortedIter::<$iter>::py_next(slf)
                }
                fn __length_hint__(&self) -> usize {
                      self.0.len()
                  }
            }
        )+
    };
}

impl_sorted_iter!(
    iter::Iter => SortedIter,
    iter::IterRev => SortedIterRev,
    iter::IterBounded => SortedIterBounded,
    iter::IterBoundedRev => SortedIterBoundedRev,
);
