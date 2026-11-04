use crate::{abc, collections::sorted::core::IterRes};
use derive_more::{Deref, From};
use pyo3::{PyClass, prelude::*};
use sorted_rs::{Bounds, ListGetter, iter};
use tap::Conv;

macro_rules! impl_sorted_iter {
    ($($iter:ty => $pyiter:ident),+ $(,)?) => {
        $(
            #[derive(From, Deref)]
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends = abc::PyoIterator)]
            pub struct $pyiter($iter);
            impl iter::PySortedIter<$iter> for $pyiter {}
            impl $pyiter {
                #[inline(always)]
                pub fn build(py: Python<'_>, value: impl Into<$iter>) -> PyResult<Bound<'_, abc::PyoIterator>> {
                    to_iter::<$iter, Self>(py, value)
                }
            }
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

#[inline(always)]
pub fn build<'py>(slf: &Bound<'py, impl ListGetter>, bounds: Bounds, rev: bool) -> IterRes<'py> {
    if rev {
        SortedIterBoundedRev::build(slf.py(), (slf, bounds))
    } else {
        SortedIterBounded::build(slf.py(), (slf, bounds))
    }
}
#[inline(always)]
fn to_iter<T, U>(py: Python<'_>, value: impl Into<T>) -> PyResult<Bound<'_, abc::PyoIterator>>
where
    U: From<T> + PyClass<BaseType = abc::PyoIterator>,
{
    value
        .conv::<T>()
        .conv::<U>()
        .into_bound(py)
        .map(Bound::into_super)
}
