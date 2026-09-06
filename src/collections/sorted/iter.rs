use std::sync::Mutex;

use crate::abc;
use crate::traits::IntoInit;
use pyo3::{PyClass, prelude::*};
use sorted_rs::iter::{Bounded, BoundedRev, Full, FullRev, ListDataIteratorMethods};
use sorted_rs::{KeysListsData, ListsData};
pub trait PySortedIter: PyClass<BaseType = abc::PyoIterator> + IntoInit {
    fn into_pyiterator(self, py: Python<'_>) -> PyResult<Bound<'_, abc::PyoIterator>> {
        self.into_bound(py).map(Bound::into_super)
    }
}
macro_rules! impl_sorted_iter {
    ($($t:ty => { $($iter:ident => $name:ident),+ $(,)? }),+ $(,)?) => {
        $($(
            #[pyclass(module = "pyochain._iterators", frozen, generic, extends=abc::PyoIterator)]
            pub struct $name(Mutex<$iter<$t>>);
            impl PySortedIter for $name {}
            impl abc::traits::ImplPyoIterator for $name {}
            impl From<$iter<$t>> for $name {
                fn from(inner: $iter<$t>) -> Self {
                    Self(Mutex::new(inner))
                }
            }
            #[pymethods]
            impl $name {
                fn __next__(&self, py: Python<'_>) -> Option<Py<PyAny>> {
                    self.0.lock().expect("poisoned").next(py)
                }
            }
        )+)+
    };
}

impl_sorted_iter! {
    ListsData => {
        Bounded => PyBounded,
        BoundedRev => PyBoundedRev,
        Full => PyFull,
        FullRev => PyFullRev,
    },
    KeysListsData => {
        Bounded => PyBoundedKey,
        BoundedRev => PyBoundedKeyRev,
        Full => PyFullKey,
        FullRev => PyFullKeyRev,
    },
}
