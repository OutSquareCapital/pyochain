use crate::{abc, collections::sorted::core::IterRes};
use derive_more::{Deref, From};
use pyo3::prelude::*;
use sorted_rs::{
    Bounds, ListGetter,
    iter::{self, PySortedIter},
};

macro_rules! impl_sorted_iter {
    ($($iter:ty => $pyiter:ident),+ $(,)?) => {
        $(
            #[derive(From, Deref)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $pyiter($iter);
            impl iter::PySortedIter for $pyiter {
                type Iter = $iter;
            }
            #[pymethods]
            impl $pyiter {
                fn __next__<'py>(slf: &Bound<'py, Self>) -> Option<Bound<'py, PyAny>> {
                    iter::PySortedIter::py_next(slf)
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

#[inline(always)]
pub fn bounded<'py>(slf: &Bound<'py, impl ListGetter>, bounds: Bounds, rev: bool) -> IterRes<'py> {
    if rev {
        SortedIterBoundedRev::build(slf.py(), (slf, bounds)).map(Bound::into_super)
    } else {
        SortedIterBounded::build(slf.py(), (slf, bounds)).map(Bound::into_super)
    }
}
